---
layout: post
title: "Bringing maximum-inner-product into Lucene"
published: true
tags: [elastic, elasticsearch, lucene]
---

> Originally published on [Elasticsearch Labs](https://www.elastic.co/search-labs/blog/lucene-bringing-maximum-inner-product-to-lucene).


## Negative values and Lucene optimizations

Lucene requires non-negative scores, so that matching one more clause in a disjunctive query can only make the score greater, not lower. This is actually important for dynamic pruning optimizations such as [block-max WAND](https://www.elastic.co/blog/faster-retrieval-of-top-hits-in-elasticsearch-with-block-max-wand), whose efficiency is largely defeated if some clauses may produce negative scores. How does this requirement affect non-normalized vectors?

In the normalized case, all vectors are on a unit sphere. This allows handling negative scores to be simple scaling.

![Normalized Vectors](/assets/lucene-bringing-maximum-inner-product-to-lucene/normalized-vectors.png)
*Figure 1: Two opposite, two dimensional vectors in a 2d unit sphere (e.g. a unit circle). When calculating the dot-product here, the worst it can be is -1 = [1, 0] * [-1, 0]. Lucene accounts for this by adding 1 to the result.*

With vectors retaining their magnitude, the range of possible values is unknown.

![Non-normalized Vectors](/assets/lucene-bringing-maximum-inner-product-to-lucene/non-normalized-vectors.png)
*Figure 2: When calculating the dot-product for these vectors `[2, 2] \* [-5, -5] = -20`*

To allow Lucene to utilize blockMax WAND with non-normalized vectors, we must scale the scores. This is a fairly simple solution. Lucene will scale non-normalize vectors with a simple piecewise function:

```java
if (dotProduct < 0) {
  return 1 / (1 + -1 * dotProduct);
}
return dotProduct + 1;
```

Now all negative scores are between 0-1, and all positives are scaled above 1. This still ensures that higher values mean better matches and removes negative scores. Simple enough, but this is not the final hurdle.

## The triangle problem

Maximum-inner-product doesn't follow the same rules as of [simple euclidean spaces](https://en.wikipedia.org/wiki/Euclidean_space). The simple assumed knowledge of the [triangle inequality](https://en.wikipedia.org/wiki/Triangle_inequality) is abandoned. Unintuitively, a vector is no longer nearest to itself. This can be troubling. Lucene's underlying index structure for vectors is Hierarchical Navigable Small World (HNSW). This being a graph based algorithm, it might rely on euclidean space assumptions. Or would exploring the graph be too slow in non-euclidean space?

Some research has indicated that a transformation into [euclidean space is required for fast search](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/XboxInnerProduct.pdf). Others have gone through the trouble of [updating their vector storage](https://blog.vespa.ai/announcing-maximum-inner-product-search/) enforcing transformations into euclidean space.

This caused us to pause and dig deep into some data. The key question is this: does HNSW provide good recall and latency with maximum-inner-product search? While the original [HNSW paper](https://arxiv.org/pdf/1603.09320.pdf) and [other published research](http://boytsov.info/pubs/thesis_boytsov.pdf) indicate that it does, we needed to do our due diligence.

## Experiments and results: Maximum-inner-product in Lucene

The experiments we ran were simple. All of the experiments are over real data sets or slightly modified real data sets. This is vital for benchmarking as modern neural networks create vectors that adhere to specific characteristics ([see discussion in section 7.8 of this paper](https://arxiv.org/pdf/1908.10396.pdf)). We measured latency (in milliseconds) vs. recall over non-normalized vectors. Comparing the numbers with the same measurements but with a euclidean space transformation. In each case, the vectors were indexed into Lucene's HNSW implementation and we measured for 1000 iterations of queries. Three individual cases were considered for each dataset: data inserted ordered by magnitude (lesser to greater), data inserted in a random order, and data inserted in reverse order (greater to lesser).

Here are some results from real datasets from Cohere:

![Cohere wiki ordered by magnitude](/assets/lucene-bringing-maximum-inner-product-to-lucene/cohere-wiki-ordered.png)
*Ordered by magnitude (lesser to greater)*

![Cohere wiki random order](/assets/lucene-bringing-maximum-inner-product-to-lucene/cohere-wiki-random.png)
*Random order*

![Cohere wiki reverse order](/assets/lucene-bringing-maximum-inner-product-to-lucene/cohere-wiki-reverse.png)

*Figure 3: Here are results for the Cohere's Multilingual model embedding wikipedia articles. [Available on HuggingFace](https://huggingface.co/datasets/Cohere/wikipedia-22-12-en-embeddings). The first 100k documents were indexed and tested.*

![Cohere multi ordered by magnitude](/assets/lucene-bringing-maximum-inner-product-to-lucene/cohere-multi-ordered.png)
*Ordered by magnitude (lesser to greater)*

![Cohere multi random order](/assets/lucene-bringing-maximum-inner-product-to-lucene/cohere-multi-random.png)
*Random order*

![Cohere multi reverse order](/assets/lucene-bringing-maximum-inner-product-to-lucene/cohere-multi-reverse.png)

*Figure 4: This is a mixture of Cohere's English and Japanese embeddings over wikipedia. [Both](https://huggingface.co/datasets/Cohere/wikipedia-22-12-en-embeddings) [datasets](https://huggingface.co/datasets/Cohere/wikipedia-22-12-ja-embeddings) are available on HuggingFace.*

We also tested against some synthetic datasets to ensure our rigor. We created a data set with [e5-small-v2](https://huggingface.co/intfloat/e5-small-v2) and scaled the vector's magnitudes by different statistical distributions. For brevity, I will only show two distributions.

![Pareto ordered by magnitude](/assets/lucene-bringing-maximum-inner-product-to-lucene/pareto-ordered.png)
*Ordered by magnitude (lesser to greater)*

![Pareto random order](/assets/lucene-bringing-maximum-inner-product-to-lucene/pareto-random.png)
*Random order*

![Pareto reverse order](/assets/lucene-bringing-maximum-inner-product-to-lucene/pareto-reverse.png)

*Figure 5: [Pareto distribution](https://en.wikipedia.org/wiki/Pareto_distribution) of magnitudes. A pareto distribution has a "fat tail" meaning there is a portion of the distribution with a much larger magnitude than others.*

![Gamma ordered by magnitude](/assets/lucene-bringing-maximum-inner-product-to-lucene/gamma-ordered.png)
*Ordered by magnitude (lesser to greater)*

![Gamma random order](/assets/lucene-bringing-maximum-inner-product-to-lucene/gamma-random.png)
*Random order*

![Gamma reverse order](/assets/lucene-bringing-maximum-inner-product-to-lucene/gamma-reverse.png)

*Figure 6: [Gamma distribution](https://en.wikipedia.org/wiki/Gamma_distribution) of magnitudes. This distribution can have high variance and makes it unique in our experiments.*

In all our experiments, the only time where the transformation seemed warranted was the synthetic dataset created with the gamma distribution. Even then, the vectors must be inserted in reverse order, largest magnitudes first, to justify the transformation. These are exceptional cases.

If you want to read about all the experiments, and about all the mistakes and improvements along the way, here is the [Lucene Github issue](https://github.com/apache/lucene/issues/12342) with all the details (and mistakes along the way). Here's one for open research and development!
