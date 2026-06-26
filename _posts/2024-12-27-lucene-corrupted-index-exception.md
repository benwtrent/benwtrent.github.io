---
layout: post
title: "Lucene bug adventures: Fixing a corrupted index exception"
published: true
tags: [elastic, elasticsearch, lucene]
---

> Originally published on [Elasticsearch Labs](https://www.elastic.co/search-labs/blog/lucene-corrupted-index-exception).

## Be prepared:

This particular blog is different than usual. It's not an explanation of a new feature or a tutorial. This is about a single line of code that took three days to write. We'll be fixing a potential Apache Lucene index corruption. Some takeaways I hope you will have:

- All flaky tests are repeatable, given enough time and the right tools
- Many layers of testing are key for robust systems. However, higher levels of tests become increasingly more difficult to debug and reproduce.
- Sleep is an excellent debugger

## How Elasticsearch tests

At Elastic, we have a plethora of tests that run against the Elasticsearch codebase. Some are simple and focused functional tests, others are single node "happy path" integration tests, and yet others attempt to break the cluster to make sure everything behaves correctly in a failure scenario. When a test continually fails, an engineer or tooling automation will create a github issue and flag it for a particular team to investigate. This [particular bug](https://github.com/elastic/elasticsearch/issues/105122) was discovered by a test of the last kind. These tests are tricky, sometimes only being repeatable after many runs.

## What is this test actually testing?

![GitHub issue for elasticsearch/issues/105122](/assets/lucene-corrupted-index-exception/github-issue-elasticsearch.jpg)

This particular test is an interesting one. It will create a particular mapping and apply it to a primary shard. Then when attempting to create a replica. The key difference is that when the replica attempts to parse the document, the test injects an exception, thus causing the recovery to fail in a surprising (but expected) way.

Everything was working as expected, however, with one significant catch. During the test cleanup, we validated consistency, and there, this test ran into a snag.

This test was failing to fail an expected manner. During the consistency check we would verify that all the replicated and primary Lucene segment files were consistent. Meaning, uncorrupted and fully replicated. Having partial data or corrupted data is way worse than having something fail fully. Here is the scary and abbreviated stack trace of the failure.

```
Caused by: org.apache.lucene.index.CorruptIndexException: Problem reading index from store(ByteSizeCachingDirectory(ElasticsearchMockDirectoryWrapper(HybridDirectory@/opt/buildkite-agent/builds/bk-agent-prod-gcp-1707109485745743789/elastic/elasticsearch-periodic/server/build/testrun/internalClusterTest/temp/org.elasticsearch.indices.recovery.IndexRecoveryIT_40853F21F419B395-001/tempDir-005/node_t0/indices/ZNwxG7VvShuwYV78RTjknA/0/index lockFactory=org.apache.lucene.store.NativeFSLockFactory@2c169f59))) (resource=store(ByteSizeCachingDirectory(ElasticsearchMockDirectoryWrapper(HybridDirectory@/opt/buildkite-agent/builds/bk-agent-prod-gcp-1707109485745743789/elastic/elasticsearch-periodic/server/build/testrun/internalClusterTest/temp/org.elasticsearch.indices.recovery.IndexRecoveryIT_40853F21F419B395-001/tempDir-005/node_t0/indices/ZNwxG7VvShuwYV78RTjknA/0/index lockFactory=org.apache.lucene.store.NativeFSLockFactory@2c169f59))))

    at org.apache.lucene.index.SegmentCoreReaders.<init>(SegmentCoreReaders.java:165)
    at org.apache.lucene.index.SegmentReader.<init>(SegmentReader.java:96)
    at org.apache.lucene.index.ReadersAndUpdates.getReader(ReadersAndUpdates.java:178)
    at org.apache.lucene.index.ReadersAndUpdates.getLatestReader(ReadersAndUpdates.java:243)
    at org.apache.lucene.index.SoftDeletesRetentionMergePolicy.keepFullyDeletedSegment(SoftDeletesRetentionMergePolicy.java:82)
    at org.apache.lucene.index.FilterMergePolicy.keepFullyDeletedSegment(FilterMergePolicy.java:118)
    at org.apache.lucene.index.FilterMergePolicy.keepFullyDeletedSegment(FilterMergePolicy.java:118)
    at org.apache.lucene.index.ReadersAndUpdates.keepFullyDeletedSegment(ReadersAndUpdates.java:822)
    at org.apache.lucene.index.IndexWriter.isFullyDeleted(IndexWriter.java:6078)
    <snip>

    Caused by: java.io.FileNotFoundException: No sub-file with id .kdi found in compound file "_0.cfs" (fileName=_0.kdi files: [_0.pos, .nvm, .fnm, _0.tip, _Lucene90_0.dvd, _0.doc, _0.tim, _Lucene90_0.dvm, _ES87BloomFilter_0.bfm, .fdm, .nvd, _ES87BloomFilter_0.bfi, _0.tmd, .fdx, .fdt])

      at org.apache.lucene.codecs.lucene90.Lucene90CompoundReader.openInput(Lucene90CompoundReader.java:170)
      at org.apache.lucene.codecs.lucene90.Lucene90PointsReader.<init>(Lucene90PointsReader.java:63)
      at org.apache.lucene.codecs.lucene90.Lucene90PointsFormat.fieldsReader(Lucene90PointsFormat.java:74)
      at org.apache.lucene.index.SegmentCoreReaders.<init>(SegmentCoreReaders.java:152)
      <snip>
```

Somehow, during the forced replication failure the replicated shard ended up getting corrupted! Let me explain the key part of the error in plain english.

Lucene is a segment based architecture, meaning each segment knows and manages its own read-only files. This particular segment was being validated via its [SegmentCoreReaders](https://github.com/apache/lucene/blob/add9c09c84ee66d4522c566c9f679035a0dfec13/lucene/core/src/java/org/apache/lucene/index/SegmentCoreReaders.java) to ensure everything was copacetic. Each core reader has metadata stored that indicates what field types and files exist for a given segment. However, when validating the [Lucene90PointsFormat](https://github.com/apache/lucene/blob/add9c09c84ee66d4522c566c9f679035a0dfec13/lucene/core/src/java/org/apache/lucene/codecs/lucene90/Lucene90PointsFormat.java), certain expected files were missing. With the segments `_0.cfs` file we expected a point format file called `kdi`. `cfs` stands for "compound file system" into which Lucene will sometimes combine all field types and all tiny files into a single larger file for more efficient replication and resource utilization. In fact, all three of the point file extensions: `kdd`, `kdi`, and `kdm` were missing. How could we get into the place where a Lucene segment expects to find a point file but it's missing!?! Seems like a scary corruption bug!

## The first step for every bug fix, replicate it

Replicating the failure for this particular bug was extremely painful. While we take advantage of [randomized value testing](https://en.wikipedia.org/wiki/Random_testing) in Elasticsearch, we are sure to provide every failure with a (hopefully) reproducible random seed to ensure all failures can be investigated. Well, this works great for all failures except for those caused by a [race condition](https://en.wikipedia.org/wiki/Race_condition).

```bash
./gradlew ':server:internalClusterTest' --tests "org.elasticsearch.indices.recovery.IndexRecoveryIT.testDoNotInfinitelyWaitForMapping" -Dtests.seed=40853F21F419B395 -Dtests.jvm.argline="-Des.concurrent_search=true" -Dtests.locale=id-ID -Dtests.timezone=Asia/Jerusalem -Druntime.java=21
```

No matter how many times I tried, the particular seed never repeated the failure locally. But, there are ways to exercise the tests and push towards a more repeatable failure.

Our particular test suite allows for a given test to be run more than once in the same command via the `-Dtests.iters` parameter. But this wasn't enough, I needed to make sure that the execution threads were switching and thus increasing the likelihood of this race condition occurring. Another wrench in the system was that the test ended up taking so long to run, the test runner would timeout. In the end, I used the following nightmare bash to repeatably run the test:

```bash
for run in {1..10}; do ./gradlew ':server:internalClusterTest' --tests "org.elasticsearch.indices.recovery.IndexRecoveryIT.testDoNotInfinitelyWaitForMapping" -Dtests.jvm.argline="-Des.concurrent_search=true" -Dtests.iters=10 ; done || exit 1
```

In comes [stress-ng](https://github.com/ColinIanKing/stress-ng). This allows you to quickly start a process that will just eat CPU cores for lunch. Randomly spamming stress-ng while running numerous iterations of the failing test finally allowed me to replicate the failure. One step closer. To stress the system, just open another terminal window and run:

```bash
stress-ng --cpu 16
```

## Revealing the bug

Now that the test failure revealing the bug is mostly repeatable, time to try and find the cause. What makes this particular test strange is that Lucene is throwing because it expects point values, but none are added directly by the test. Only text values. This pushed me to consider looking at recent changes to our [optimistic concurrency control](https://www.elastic.co/guide/en/elasticsearch/reference/current/optimistic-concurrency-control.html) fields: `_seq_no` and `_primary_term`. Both of these are indexed as points and exist in every Elasticsearch document.

Indeed a [commit](https://github.com/elastic/elasticsearch/pull/105036) did change our `_seq_no` mapper! YES! This has to be the cause! But, my excitement was short-lived. This only changed the order of when fields got added to the document. Before this change, `_seq_no` fields were added last to the document. After, they were added first. No way the order of adding fields to a Lucene document would cause this failure...

Yep, changing the order of when fields were added caused the failure. This was surprising and turns out to be a bug in Lucene itself! Changing the order of what fields are parsed shouldn't change the behavior of parsing a document.

## The bug in Lucene

Indeed, the bug in Lucene focused on following conditions:

- Indexing a points value field (e.g. `_seq_no`)
- Trying to index a text field throw during analysis
- In this weird state, we open a [Near Real Time Reader](https://blog.mikemccandless.com/2011/06/lucenes-near-real-time-search-is-fast.html) from the writer that experienced the text index analysis exception

But no matter how many ways I tried, I couldn't fully replicate. I directly added pause points for debugging throughout the Lucene codebase. I attempted randomly opening readers during the exception path. I even printed out megabytes and megabytes of logs trying to find the exact path where this failure occurred. I just couldn't do it. I spent a whole day fighting and losing.

Then I slept.

The next day I re-read the original stack trace and discovered the following line:

```
at org.apache.lucene.index.SoftDeletesRetentionMergePolicy.keepFullyDeletedSegment(SoftDeletesRetentionMergePolicy.java:82)
```

In all my recreation attempts, I never specifically set the retention merge policy. The [SoftDeletesRetentionMergePolicy](https://github.com/apache/lucene/blob/5f0fa2b291ff9e7d878642f025a70c15b788a470/lucene/core/src/java/org/apache/lucene/index/SoftDeletesRetentionMergePolicy.java) is used by Elasticsearch so that we can accurately replicate deletions in replicas and ensure all our concurrency controls are in charge of when documents are actually removed. Otherwise, Lucene is in full control and will remove them at any merge.

Once I added this policy and replicated the most basic steps mentioned above, the failure immediately replicated.

I have never been more happy to open a [bug in Lucene](https://github.com/apache/lucene/issues/13353).

![GitHub issue for apache/lucene/issues/13353](/assets/lucene-corrupted-index-exception/github-issue-lucene.jpg)

While it presented itself as a race condition in Elasticsearch, it was simple to write a repeatably failing test in Lucene once all the conditions were met.

In the end, like all good bugs, it was fixed with just 1 line of code. Multiple days of work, for just one line of code.

![One line of code fix](/assets/lucene-corrupted-index-exception/one-line-fix.jpg)

But it was worth it.
