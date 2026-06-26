---
layout: post
title: "Filtered HNSW search, fast mode"
published: true
tags: [elastic, elasticsearch, lucene, vector]
---

> Originally published on [Elasticsearch Labs](https://www.elastic.co/search-labs/blog/filtered-hnsw-knn-search).

## Why searching fewer docs is actually slower in HNSW

Counterintuitively, filtering documents—thereby reducing the number of candidates—can actually make kNN searches slower. For traditional lexical search, fewer documents mean fewer scoring operations, meaning faster search. However, in an HNSW graph, the primary cost is the number of **vector comparisons** needed to identify the k nearest neighbors. At certain filter set sizes, the number of vector comparisons can increase significantly, slowing down search performance.

![Unfiltered HNSW graph search](/assets/filtered-hnsw-knn-search/unfiltered-hnsw-search.gif)
*Here is an example of unfiltered graph search. Note there are about 6 vector operations.*

Because the HNSW graph in Apache Lucene has no knowledge of filtering criteria when built, it constructs purely based on vector similarity. When applying a filter to retrieve the k nearest neighbors, the search process traverses more of the graph. This happens because the natural nearest neighbors within a local graph neighborhood may be **filtered out**, requiring deeper exploration and increasing the number of vector comparisons.

![Filtered HNSW graph search example](/assets/filtered-hnsw-knn-search/filtered-hnsw-search.gif)
*Here is an example of the current filtered graph search. The "dashed circles" are vectors that do not match the filter. We even make vector comparisons against the filtered out vectors, resulting in more vector ops, about 9 total.*

You may ask, why perform vector comparisons against nodes that don't match the filter at all? Well, HNSW graphs are already sparsely connected. If we were to consider only matching nodes during exploration, the search process could easily **get stuck**, unable to traverse the graph efficiently.

![Typical HNSW graph with filtered gap](/assets/filtered-hnsw-knn-search/hnsw-graph-filtered-gap.png)
*Note the filtered "gulf" between the entry point and the first valid filtered set. In a typical graph, it's possible for such a gap to exist, causing exploration to end prematurely and resulting in poor recall.*

## We gotta make this faster: Improving HNSW vector search in Lucene

Since the graph doesn't account for filtering criteria, we have to explore the graph more. Additionally, to avoid getting stuck, we must perform vector comparisons against filtered-out nodes. How can we reduce the number of vector operations without getting stuck? This is the exact problem tackled by [Liana Patel et. al. in their ACORN](https://arxiv.org/abs/2403.04871) paper.

While the paper discusses multiple graph techniques, the specific algorithm we care about with Apache Lucene is their ACORN-1 algorithm. The main idea is that you only explore nodes that satisfy your filter. To compensate for the increased sparsity, ACORN-1 extends the exploration beyond the immediate neighborhood. Now, instead of exploring just the immediate neighbors, each neighbor's neighbor is also explored. This means that for a graph with 32 connections, instead of only looking at the nearest 32 neighbors, exploration will attempt to find matching neighbors in 32*32=1024 extended neighborhood.

![ACORN algorithm in action](/assets/filtered-hnsw-knn-search/acorn-algorithm.gif)
*Here you can see the ACORN algorithm in action. Only doing vector comparisons and exploration for valid matching vectors, quickly expanding from the immediate neighborhood. Resulting in much fewer vector ops, about 6 in total.*

Within Lucene, we have slightly adapted the ACORN-1 algorithm in the following ways. The extended neighborhoods are only explored if more than 10% of the vectors are filtered out in the immediate neighborhood. Additionally, the extended neighborhood isn't explored if we have already scored at least `neighborCount * 1.0/(1.0 - neighborFilterRatio)`. This allows the searcher to take advantage of more densely connected neighborhoods where the neighborhood connectedness is highly correlated with the filter.

We also have noticed both in inversely correlated filters (e.g. filters that only match vectors that are far away from the query vector) or exceptionally restrictive filters, only exploring the neighborhood of each neighbor isn't enough. The searcher will also attempt branching further than the neighbors' neighbors when no valid vectors passing the filter are found. However, to prevent getting lost in the graph, this additional exploration is bounded.

## Numbers don't lie

Across multiple real-world datasets, this new filtering approach has delivered **significant speed improvements**. Here is randomly filtering at 0.05% [1M Cohere vectors](https://huggingface.co/datasets/Cohere/wikipedia-2023-11-embed-multilingual-v3):

![1M Cohere vectors benchmark](/assets/filtered-hnsw-knn-search/cohere-1m-benchmark.png)
*Up and to the left is "winning", which shows that the candidate is significantly better. Though, to achieve the same recall, search parameters (e.g. `num_candidates`) need to be adjusted.*

To further investigate this reduction in improvement as more vectors pass the filter, we did another test over an [8M Cohere Wiki document data set](https://huggingface.co/datasets/Cohere/wikipedia-22-12-en-embeddings). Generally, no matter the number of vectors filtered, you want higher recall, with fewer visited vectors. A simple way to quantify this is by examining the **recall-to-visited ratio**.

![Recall vs visited ratio](/assets/filtered-hnsw-knn-search/recall-vs-visited-ratio.png)
*Here we see how the new filtered search methodology achieves much better recall vs. visited ratio.*

It's clear that near 60%, the improvements level off or disappear. Consequently, in Lucene, this new algorithm will only be utilized when 40% or more of the vectors are filtered out.

Even our nightly Lucene benchmarks saw an impressive improvement with this change.

![Apache Lucene nightly benchmark results](/assets/filtered-hnsw-knn-search/lucene-nightly-benchmark.png)
*Apache Lucene runs over 8M 768 document vectors with a random filter that allows 5% of the vectors to pass. These kinds of graphs make me happy.*
