---
layout: post
title: "Optimized Scalar Quantization: Improving Better Binary Quantization (BBQ)"
published: true
tags: [elastic, elasticsearch]
---

> Originally published on [Elasticsearch Labs](https://www.elastic.co/search-labs/blog/optimized-scalar-quantization-elasticsearch).

## Scalar quantization history

Introduced in Elasticsearch 8.12, [scalar quantization](https://www.elastic.co/search-labs/blog/scalar-quantization-101) was initially a simple min/max quantization scheme. Per lucene segment, we would find the global quantile values for a given confidence interval. These quantiles are then used as the minimum and maximum to quantize all the vectors. While this naive quantization is powerful, it only really works for whole byte quantization.

![Scalar quantization: Static Confidence Intervals](/assets/optimized-scalar-quantization-elasticsearch/static-confidence-intervals.png)
*Static confidence intervals mean static quantiles. This is calculated once for all vectors in a given segment and works well for higher bit values.*

In Elasticsearch 8.15, we added [half-byte, or int4, quantization](https://www.elastic.co/search-labs/blog/int4-scalar-quantization-in-lucene). To achieve this with high recall, we added an optimization step, allowing for the best quantiles to be calculated dynamically. Meaning, no more static confidence intervals. Lucene will calculate the best global upper and lower quantiles for each segment. Achieving 8x reduction in memory utilization over float32 vectors.

![Scalar quantization: reducing the vector similarity error](/assets/optimized-scalar-quantization-elasticsearch/dynamic-quantile-optimization.gif)
*Dynamically searching for the best quantiles to reduce the vector similarity error. This was done once, globally, over a sample set of the vectors and applied to all.*

Finally, now in 8.18, we have added locally optimized scalar quantization. It optimizes quantiles per individual vector. Allowing for exceptional recall at any bit size, even single bit quantization.

## What is Optimized Scalar Quantization?

For an in-depth explanation of the math and intuition behind optimized scalar quantization, check out our blog post on [Optimized Scalar Quantization](https://www.elastic.co/search-labs/blog/scalar-quantization-optimization). There are three main takeaways from this work:

- Each vector, is centered on the Apache Lucene segment's centroid. This allows us to make better use of the possible quantized vectors to represent the dataset as a whole.
- Every vector is individually quantized with a unique set of optimized quantiles.
- Asymmetric quantization is used allowing for higher recall with the same memory footprint.

In short, when quantizing each vector:

- We center the vector on the centroid
- Compute a limited number of iterations to find the optimal quantiles. Stopping early if the quantiles are unchanged or the error (loss) increases
- Pack the resulting quantized vectors
- Store the packed vector, its quantiles, the sum of its components, and an extra error correction term

![Optimization Steps: Scalar optimization](/assets/optimized-scalar-quantization-elasticsearch/optimization-steps.gif)
*Here is a step by step view of optimizing 2 bit vectors. After the fourth iteration, we would normally stop the optimization process as the error (loss) increased. The first cell is each individual components error. The second is the distribution of 2 bit quantized vectors. Third is how the overall error is changing. Fourth is current step's quantiles overlayed of the raw vector being quantized.*

## Storage and retrieval of optimized scalar quantization

The storage and retrieval of optimized scalar quantization vectors are similar to BBQ. The main difference is the particular values we store.

![Storage and retrieval of optimized scalar quantization](/assets/optimized-scalar-quantization-elasticsearch/storage-retrieval.png)
*Stored for every binary quantized vector: dims/8 bytes, upper and lower quantiles, an additional correction term, the sum of the quantized components.*

One piece of nuance is the correction term. For [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance), we store the [squared norm](https://en.wikipedia.org/wiki/Norm_(mathematics)) of the centered vector. For [dot product](https://en.wikipedia.org/wiki/Dot_product) we store the dot product between the centroid and the uncentered vector.

## Performance

Enough talk. Here are the results from four datasets.

- [Cohere's 768 dimensioned](https://huggingface.co/datasets/Cohere/wikipedia-22-12-simple-embeddings) multi-lingual embeddings. This is a well distributed inner-product dataset.
- [Cohere's 1024 dimensioned](https://huggingface.co/datasets/Cohere/wikipedia-2023-11-embed-multilingual-v3) multi-lingual embeddings. This embedding model is well optimized for quantization.
- [E5-Small-v2](https://huggingface.co/intfloat/e5-small-v2) quantized over the [quora](https://www.kaggle.com/c/quora-question-pairs) dataset. This model typically does poorly with binary quantization.
- [GIST-1M](https://corpus-texmex.irisa.fr/) dataset. This scientific dataset opens some interesting edge cases for inner-product and quantization.

Here are the results for *Recall@10\|50*

| Dataset | BBQ | BBQ with OSQ | Improvement |
|---------|-----|--------------|-------------|
| Cohere 768 | 0.933 | 0.938 | 0.5% |
| Cohere 1024 | 0.932 | 0.945 | 1.3% |
| E5-Small-v2 | 0.972 | 0.975 | 0.3% |
| GIST-1M | 0.740 | 0.989 | 24.9% |

Across the board, we see that BBQ backed by our new optimized scalar quantization improves recall, and dramatically so for the GIST-1M dataset.

But, what about indexing times? Surely all this per vector optimizations must add up. The answer is no.

Here are the indexing times for the same datasets.

| Dataset | BBQ | BBQ with OSQ | Difference |
|---------|-----|--------------|------------|
| Cohere 768 | 368.62s | 372.95s | +1% |
| Cohere 1024 | 307.09s | 314.08s | +2% |
| E5-Small-v2 | 227.37s | 229.83s | < +1% |
| GIST-1M | 1300.03s* | 297.13s | -300% |

- Since the quantization methodology works so poorly over GIST-1M when using inner-product, it takes an exceptionally long time to build the HNSW graph as the vector distances are not well distinguished.
