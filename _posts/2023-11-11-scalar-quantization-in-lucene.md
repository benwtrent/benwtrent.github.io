---
layout: post
title: "Understanding scalar quantization in Lucene"
published: true
use_math: true
tags: [elastic, elasticsearch, lucene]
---

> Originally published on [Elasticsearch Labs](https://www.elastic.co/search-labs/blog/scalar-quantization-in-lucene).

## Automatic byte quantization in Lucene

While HNSW is a powerful and flexible way to store and search vectors, it does require a significant amount of memory to run quickly. For example, querying 1MM float32 vectors of 768 dimensions requires roughly $1,000,000 * 4 * (768 + 12) = 3120000000 bytes \approx 3GB$ of ram. Once you start searching a significant number of vectors, this gets expensive. One way to use around $75\%$ less memory is through byte quantization. Lucene and consequently Elasticsearch has supported indexing $byte$ vectors for some time, but building these vectors has been the user's responsibility. This is about to change, as we have introduced $int8$ scalar quantization in Lucene.

## Scalar quantization 101

All quantization techniques are considered lossy transformations of the raw data. Meaning some information is lost for the sake of space. For an in depth explanation of scalar quantization, see: [Scalar Quantization 101](https://www.elastic.co/search-labs/scalar-quantization-101). At a high level, scalar quantization is a lossy compression technique. Some simple math gives significant space savings with very little impact on recall.

## Exploring the architecture

Those used to working with Elasticsearch may be familiar with these concepts already, but here is a quick overview of the distribution of documents for search.

Each Elasticsearch index is composed of [multiple shards](https://www.elastic.co/guide/en/elasticsearch/reference/current/size-your-shards.html#size-your-shards). While each shard can only be assigned to a single node, multiple shards per index gives you compute parallelism across nodes.

Each shard is composed as a single [Lucene Index](https://lucene.apache.org/core/9_8_0/core/org/apache/lucene/index/package-summary.html). A Lucene index consists of multiple read-only segments. During indexing, documents are buffered and periodically flushed into a read-only segment. When certain conditions are met, these segments can be merged in the background into a larger segment. All of this is configurable and has its own set of complexities. But, when we talk about segments and merging, we are talking about read-only Lucene segments and the automatic periodic merging of these segments. [Here is a deeper dive](https://www.elastic.co/search-labs/vector-search-elasticsearch-rationale) into segment merging and design decisions.

## Quantization per segment in Lucene

Every segment in Lucene stores the following: the individual vectors, the HNSW graph indices, the quantized vectors, and the calculated quantiles. For brevity's sake, we will focus on how Lucene stores quantized and raw vectors. For every segment, we keep track of the raw vectors in the $vec$ file, quantized vectors and a single corrective multiplier float in $veq$, and the metadata around the quantization within the $vemq$ file.

![The .vec File](/assets/scalar-quantization-in-lucene/vec-file-layout.png)
*Figure 1: Simplified layout of raw vector storage file. Takes up $dimension * 4 * numVectors$ of disk space since $float$ values are 4 bytes. Because we are quantizing, these will not get loaded during HNSW search. They are only used if specifically requested (e.g. brute-force secondary via [rescore](https://www.elastic.co/guide/en/elasticsearch/reference/current/filter-search-results.html#query-rescorer)), or for re-quantization during segment merge.*

![The .veq file](/assets/scalar-quantization-in-lucene/veq-file-layout.png)
*Figure 2: Simplified layout of the $.veq$ file. Takes up $(dimension + 4)*numVectors$ of space and will be loaded into memory during search. The $+ 4$ bytes is to account for the corrective multiplier float, used to adjust scoring for better accuracy and recall.*

![The .vemq file](/assets/scalar-quantization-in-lucene/vemq-file-layout.png)
*Figure 3: The simplified layout of the metadata file. Here is where we keep track of quantization and vector configuration along with the calculated quantiles for this segment.*

So, for each segment, we store not only the quantized vectors, but the quantiles used in making these quantized vectors and the original raw vectors. But, why do we keep the raw vectors around at all?

## Quantization that grows with you

Since Lucene periodically flushes to read only segments, each segment only has a partial view of all your data. This means the quantiles calculated only directly apply for that sample set of your entire data. Now, this isn't a big deal if your sample adequately represents your entire corpus. But Lucene allows you to sort your index in various ways. So, you could be indexing data sorted in a way that adds bias for per-segment quantile calculations. Also, you can flush the data whenever you like! Your sample set could be tiny, even just one vector. Yet another wrench is that you have control over when merges occur. While Elasticsearch has configured defaults and periodic merging, you can ask for a merge whenever you like via [_force_merge](https://www.elastic.co/guide/en/elasticsearch/reference/8.10/indices-forcemerge.html) API. So how do we still allow all this flexibility, while providing good quantization that provides good recall?

Lucene's vector quantization will automatically adjust over time. Because Lucene is designed with a read-only segment architecture, we have guarantees that the data in each segment hasn't changed and clear demarcations in the code for when things can be updated. This means during segment merge we can adjust quantiles as necessary and possibly re-quantize vectors.

![Multiple segment quantiles](/assets/scalar-quantization-in-lucene/multiple-segment-quantiles.png)
*Figure 4: Three example segments with different quantiles.*

But isn't re-quantization expensive? It does have some overhead, but Lucene handles quantiles intelligently, and only fully-requantizes when necessary. Let's use the segments in Figure 4 as an example. Let's give segments $A$ and $B$ $1,000$ documents each and segment $C$ only $100$ documents. Lucene will take a weighted average of the quantiles and if that resulting merged quantile is near enough to the segments original quantiles, we don't have to re-quantize that segment and will utilize the newly merged quantiles.

![Merged quantiles](/assets/scalar-quantization-in-lucene/merged-quantiles.png)
*Figure 5: Example of merged quantiles where segments $A$ and $B$ have $1000$ documents and $C$ only has $100$.*

In the situation visualized in figure 5, we can see that the resulting merged quantiles are very similar to the original quantiles in $A$ and $B$. Thus, they do not justify quantizing the vectors. Segment $C$, seems to deviate too much. Consequently, the vectors in $C$ would get re-quantized with the newly merged quantile values.

There are indeed extreme cases where the merged quantiles differ dramatically from any of the original quantiles. In this case, we will take a sample from each segment and fully re-calculate the quantiles.

## Quantization performance & numbers

So, is it fast and does it still provide good recall? The following numbers were gathered running the experiment on a `c3-standard-8` GCP instance. To ensure a fair comparison with $float32$ we used an instance large enough to hold raw vectors in memory. We indexed $400,000$ [Cohere Wiki](https://huggingface.co/datasets/Cohere/wikipedia-22-12-simple-embeddings) vectors using maximum-inner-product.

![Quantization Recall](/assets/scalar-quantization-in-lucene/quantization-recall.png)
*Figure 6: Recall@10 for quantized vectors vs raw vectors. The search performance of quantized vectors is significantly faster than raw, and recall is quickly recoverable by gathering just 5 more vectors; visible by $quantized@15$.*

Figure 6 shows the story. While there is a recall difference, as to be expected, it's not significant. And, the recall difference dissappears by gathering just 5 more vectors. All this with $2\times$ faster segment merges and 1/4 of the memory of $float32$ vectors.

