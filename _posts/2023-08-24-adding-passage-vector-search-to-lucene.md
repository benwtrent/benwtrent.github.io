---
layout: post
title: "Adding passage vector search to Lucene"
published: true
tags: [elastic, elasticsearch, lucene, vector]
---

> Originally published on [Elasticsearch Labs](https://www.elastic.co/search-labs/blog/adding-passage-vector-search-to-lucene).

Where lexical search like BM25 is already designed for long documents, text embedding models are not. All embedding models have limitations on the number of tokens they can embed. So, for longer text input it must be chunked into passages shorter than the model's limit. Now instead of having one document with all its metadata, you have multiple passages and embeddings. And if you want to preserve your metadata, it must be added to every new document.

![Passages illustration](/assets/adding-passage-vector-search-to-lucene/passages-illustration.png)
*Figure 1: Now instead of having a single piece of metadata indicating the first chapter of Little Women, you have to index that information data for every sentence.*

A way to address this is with Lucene's "join" functionality. This is an integral part of Elasticsearch's [nested](https://www.elastic.co/guide/en/elasticsearch/reference/current/nested.html) field type. It makes it possible to have a top-level document with multiple nested documents, allowing you to search over nested documents and join back against their parent documents. This sounds perfect for multiple passages and vectors belonging to a single top-level document! This is all awesome! But, wait, Elasticsearch doesn't support vectors in nested fields. Why not, and what needs to change?

## The (kNN) problem with parents and children

The key issue is how Lucene can join back to the parent documents when searching child vector passages. Like with [kNN pre-filtering versus post-filtering](https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html#knn-search-filter-example), when the joining occurs determines the result quality and quantity. If a user searches for the top four nearest *parent documents (not passages) to a query vector*, they usually expect four documents. But what if they are searching over child vector passages and all four of the nearest vectors are from the same parent document? This would end up returning just *one* parent document, which would be surprising. This same kind of issue occurs with post-filtering.

![Nested document structure](/assets/adding-passage-vector-search-to-lucene/nested-doc-structure.png)
*Figure 2: Documents 3, 5, 10 are parent docs. 1, 2 belong to 3; 4 to 5; 6, 7, 8, 9 to 10.*

Let us search with query vector A, and the four nearest passage vectors are 6, 7, 8, 9. With "post-joining," you only end up retrieving parent document 10.

![Vector A matching nearest children of document 10](/assets/adding-passage-vector-search-to-lucene/vector-a-matching.png)
*Figure 3: Vector "A" matching nearest all the children of 10.*

What can we do about this problem? One answer could be, "Just increase the number of vectors returned!" However, at scale, this is untenable. What if every parent has at least 100 children and you want the top 1,000 nearest neighbors? That means you have to search for at least 100,000 children! This gets out of hand quickly. So, what's another solution?

## Pre-joining to the rescue

The solution to the "post-joining" problem is "pre-joining." Recently added [changes to Lucene](https://github.com/apache/lucene/pull/12434) enable joining against the parent document while searching the HNSW graph! Like with [kNN pre-filtering](https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html#knn-search-filter-example), this ensures that when asked to find the k nearest neighbors of a query vector, we can return not the k nearest passages as represented by dense vectors, but k nearest *documents*, as represented by their child passages that are most similar to the query vector. What does this actually look like in practice?

Let's assume we are searching the same nested documents as before:

![Nested document structure](/assets/adding-passage-vector-search-to-lucene/nested-doc-structure.png)
*Figure 4: Documents 3, 5, 10 are parent docs. 1,2 belong to 3; 4 to 5; 6, 7, 8, 9 to 10.*

As we search and score documents, instead of tracking children, we track the parent documents and update their scores. Figure 5 shows a simple flow. For each child document visited, we get its score and then track it by its parent document ID. This way, as we search and score the vectors we only gather the parent IDs. This ensures diversification of results with no added complexity to the HNSW algorithm using already existing and powerful tools within Lucene. All this with only a single additional bit of memory required per vector stored.

![Figure 5: parent document scoring during HNSW traversal](/assets/adding-passage-vector-search-to-lucene/anim.gif)
*Figure 5: As we search the vectors, we score and collect the associated parent document. Only updating the score if it is more competitive than the previous.*

But, how is this efficient? Glad you asked! There are certain restrictions that provide some really nice short cuts. As you can tell from the previous examples, all parent document IDs are larger than child IDs. Additionally, parent documents do not contain vectors themselves, meaning children and parents are [purely disjoint sets](https://en.wikipedia.org/wiki/Disjoint_sets). This affords some nice optimizations via [bit sets](https://en.wikipedia.org/wiki/Bit_array). A bit set provides an exceptionally fast structure for "tell me the next bit that is set." For any child document, we can ask the bit set, "Hey, what's the number that is larger than me that is in the set?" Since the sets are disjoint, we know the next bit that is set is the parent document ID.
