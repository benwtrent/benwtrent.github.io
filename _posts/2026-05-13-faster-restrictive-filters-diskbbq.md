---
layout: post
title: "Narrow prefiltering for partitioned vector indices"
published: true
tags: [elastic, elasticsearch]
---

> Originally published on [Elasticsearch Labs](https://www.elastic.co/search-labs/blog/faster-restrictive-filters-diskbbq).

## What's hard about filtering partition indices

With partition indices, all searches are done in two phases:

- Find the nearest centroids.
- Find all the nearest vectors within the nearest centroids' clusters.

For DiskBBQ, the centroids are quantized and possibly indexed in their own structure. The cluster contents (we'll call them *postings lists* from now on) are laid out in an effort to make scoring the vectors fast. Postings are stored in blocks of 32, with each block in doc_id order. Doc IDs are delta-encoded to minimize disk usage with low decode overhead. Vector values are also block-encoded, separating dimensions from quantized corrections to maximize [single instruction, multiple data (SIMD) throughput using our optimized kernels](https://www.elastic.co/search-labs/blog/elasticsearch-vector-search-simdvec-engine).

```
Vector Cluster (Posting List) layout
  | metadata |
  | doc_deltas[32] | vec_quant[32] | vec_quant_corrections[32] |
  | doc_deltas[32] | vec_quant[32] | vec_quant_corrections[32] |
  | ... |
  | doc_deltas[T]  | vec_quant[T]  | vec_quant_corrections[T]  |   (T <= 32)
```

Once we reach a postings list, the layout is optimized for fast scoring and filtering. For example, if an index sort is provided, blocks of vectors that match a filter will be stored and scored together within a list. This unlocks scoring contiguous blocks of vectors at a time, taking full advantage of the underlying CPU throughput.

That said, we don't know if a cluster matches a given filter until we actually check its doc_ids. Once verified, we can be sure to only score against the relevant vectors. In restrictive-filter cases, we can inspect a centroid and still find that none of its vectors match the filter. To compensate, we keep scoring and exploring centroids until we get a representative group of vectors scored.

This meant wasted work for restrictive filters. We score centroids, not knowing if they have vectors relevant to the filter or not; prepare to score the postings list, only to find that none of the blocks apply. The wasted compute adds up:

- Unnecessarily scoring the centroid.
- Loading a filtered-out postings list because it's close to the query vector.
- Decoding and checking the document IDs in the list, only to find out none match.
- Continue the search, potentially hitting another completely filtered-out centroid, until we visit enough to get the desired recall.

Here's an example showing the old flow. Check all the centroids, see what matches, move on to centroids that have matching vectors. Rinse and repeat.

<video autoplay loop muted playsinline style="width:100%;border-radius:4px;">
  <source src="/assets/faster-restrictive-filters-diskbbq/IVFMultiClusterFilterSearch.mp4" type="video/mp4">
</video>
<p><em>Two-dimensional DiskBBQ partitioned search showing unnecessary work when centroids are filtered out.</em></p>

## How do we get to the right centroids quickly?

The simplest solution would be to skip centroids that contain no valid vectors. But we don't want to index additional information for all potential filter fields and values. A user can provide many variations of complex or simple filters. This is a strength of Elasticsearch, and we don't want to hamper that.

Instead, we simply store the mapping of `doc_id -> centroid_ord`. This gives us an immediate view of all docs and their centroid membership. Allowing us to iterate any provided filter in document order, quickly determining the relevant centroids. Of course, iterating every document to check if it passes a filter is not free. We only apply this eager logic if the average number of documents per cluster that match the filter is `1.25`. Yes, this is a "magic number"; however, it's empirically based. Assuming the filter is random, we're validating at least one matching vector per centroid with some overhead. We may refine this in the future, but early experimentation found this number to be a sweet spot for most users.

<video autoplay loop muted playsinline style="width:100%;border-radius:4px;">
  <source src="/assets/faster-restrictive-filters-diskbbq/IVFFilterOnlySearch.mp4" type="video/mp4">
</video>
<p><em>The new algorithm going straight to centroids that match the provided filter.</em></p>

Here's the new way. Detecting we have a restrictive filter, go straight to the filtered centroids.

## Benchmark, benchmark, benchmark

Here's a macro-benchmark with a random filter. The filter selectivity is purposefully extreme to show the significant improvement on hyper-restrictive filters. Here we see almost an order-of-magnitude improvement. Where before, when filters got very restrictive, there would be a horrible elbow. Now latency remains consistent and will in fact improve as filters get more restrictive.

![Line chart comparing latency vs filter selectivity, showing 3–5x improvement with the new approach](/assets/faster-restrictive-filters-diskbbq/filter-search-benchmark.png)

A further validation is our nightly runs of [so-vector](https://github.com/elastic/rally-tracks/tree/master/so_vector) with [rally](https://github.com/elastic/rally) showing the improvement. You can try this yourself by specifying `bbq_disk` in the `vector_index_type` in the rally configuration.

![Line and area chart showing latency in milliseconds over time, with values decreasing sharply in late November 2025](/assets/faster-restrictive-filters-diskbbq/rally-latency.png)