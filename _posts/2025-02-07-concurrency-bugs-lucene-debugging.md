---
layout: post
title: "Squashing a concurrency bug in Apache Lucene"
published: true
tags: [elastic, elasticsearch, lucene]
---

> Originally published on [Elasticsearch Labs](https://www.elastic.co/search-labs/blog/concurrency-bugs-lucene-debugging).

## Concurrency bugs: software engineers' bane

Concurrency bugs are the worst. Not only are they difficult to fix, simply getting them to fail reliably is the hardest part. Take this test failure, [`TestIDVersionPostingsFormat#testGlobalVersions`](https://github.com/apache/lucene/issues/13127), as an example. It spawns multiple document writing and updating threads, challenging Lucene's optimistic concurrency model. This test exposed a race condition in the optimistic concurrency control. Meaning, a document operation may falsely claim to be the latest in a sequence of operations. Meaning, in certain conditions, an update or delete operation might actually succeed when it should have failed given optimistic concurrency constraints.

```
org.apache.lucene.sandbox.codecs.idversion.TestIDVersionPostingsFormat > testGlobalVersions FAILED
    java.lang.AssertionError: maxSeqNo must be greater or equal to 7442 but was 7441
        at __randomizedtesting.SeedInfo.seed([B97A2BDBC7E40BF6:B4D76006D5101E6]:0)
        at org.apache.lucene.core@10.0.0-SNAPSHOT/org.apache.lucene.index.DocumentsWriterDeleteQueue.close(DocumentsWriterDeleteQueue.java:325)
        at org.apache.lucene.core@10.0.0-SNAPSHOT/org.apache.lucene.index.DocumentsWriter.flushAllThreads(DocumentsWriter.java:659)
        at org.apache.lucene.core@10.0.0-SNAPSHOT/org.apache.lucene.index.IndexWriter.getReader(IndexWriter.java:576)
        at org.apache.lucene.core@10.0.0-SNAPSHOT/org.apache.lucene.index.StandardDirectoryReader.doOpenFromWriter(StandardDirectoryReader.java:381)
        at org.apache.lucene.core@10.0.0-SNAPSHOT/org.apache.lucene.index.StandardDirectoryReader.doOpenIfChanged(StandardDirectoryReader.java:355)
        at org.apache.lucene.core@10.0.0-SNAPSHOT/org.apache.lucene.index.StandardDirectoryReader.doOpenIfChanged(StandardDirectoryReader.java:345)
        at org.apache.lucene.core@10.0.0-SNAPSHOT/org.apache.lucene.index.DirectoryReader.openIfChanged(DirectoryReader.java:170)
        at org.apache.lucene.core@10.0.0-SNAPSHOT/org.apache.lucene.search.SearcherManager.refreshIfNeeded(SearcherManager.java:144)
        at org.apache.lucene.core@10.0.0-SNAPSHOT/org.apache.lucene.search.SearcherManager.refreshIfNeeded(SearcherManager.java:52)
        at org.apache.lucene.core@10.0.0-SNAPSHOT/org.apache.lucene.search.ReferenceManager.doMaybeRefresh(ReferenceManager.java:167)
        at org.apache.lucene.core@10.0.0-SNAPSHOT/org.apache.lucene.search.ReferenceManager.maybeRefresh(ReferenceManager.java:213)
```

Apologies for those who hate Java stack traces. Note, delete doesn't necessarily mean "delete". It can also indicate a document "update", as Lucene's segments are read-only.

Apache Lucene manages each thread that is writing documents through the [`DocumentsWriter`](https://github.com/apache/lucene/blob/f2e7ae40af0b28b1d5f2edc31f8858229a8523f4/lucene/core/src/java/org/apache/lucene/index/DocumentsWriter.java) class. This class will create or reuse threads for document writing and each write action controls its information within the [`DocumentsWriterPerThread`](https://github.com/apache/lucene/blob/f2e7ae40af0b28b1d5f2edc31f8858229a8523f4/lucene/core/src/java/org/apache/lucene/index/DocumentsWriterPerThread.java) (DWPT) class. Additionally, the writer keeps track of what documents are deleted in the [`DocumentsWriterDeleteQueue`](https://github.com/apache/lucene/blob/f2e7ae40af0b28b1d5f2edc31f8858229a8523f4/lucene/core/src/java/org/apache/lucene/index/DocumentsWriterDeleteQueue.java) (DWDQ). These structures keep all document mutation actions in memory and will periodically flush, freeing up in-memory resources and persisting structures to disk.

In an effort to prevent [blocking threads](https://en.wikipedia.org/wiki/Blocking_(computing)) and ensuring high throughput in concurrent systems, Apache Lucene tries to only [synchronize](https://docs.oracle.com/javase/tutorial/essential/concurrency/syncmeth.html) in very critical sections. While this can be good in practice, like in any concurrent systems, there are dragons.

## A false hope

My initial investigation pointed me to a couple of critical sections that were not appropriately synchronized. All interactions to a given [`DocumentsWriterDeleteQueue`](https://github.com/apache/lucene/blob/f2e7ae40af0b28b1d5f2edc31f8858229a8523f4/lucene/core/src/java/org/apache/lucene/index/DocumentsWriterDeleteQueue.java) are controlled by its enclosing [`DocumentsWriter`](https://github.com/apache/lucene/blob/f2e7ae40af0b28b1d5f2edc31f8858229a8523f4/lucene/core/src/java/org/apache/lucene/index/DocumentsWriter.java). So while individual methods may not be appropriately synchronized in the [`DocumentsWriterDeleteQueue`](https://github.com/apache/lucene/blob/f2e7ae40af0b28b1d5f2edc31f8858229a8523f4/lucene/core/src/java/org/apache/lucene/index/DocumentsWriterDeleteQueue.java), their access to the world is (or should be). (Let's not delve into how this muddles ownership and access—it's a long-lived project written by many contributors. Cut it some slack.)

However, I found [one place during a flush](https://github.com/apache/lucene/blob/40060f8b7080d06a218518445a0a1dfc520c812a/lucene/core/src/java/org/apache/lucene/index/DocumentsWriterFlushControl.java#L572-L576) that was not synchronized.

```java
// Advance the queue, meaning create a new one to keep track of deletes
// Since we have been flushed, let's starting tracking again
DocumentsWriterDeleteQueue newQueue = documentsWriter.deleteQueue.advanceQueue(perThreadPool.size());
// OK, get the current new maximum sequence number for optimistic concurrency
seqNo = documentsWriter.deleteQueue.getMaxSeqNo();
// Reset to the new queue
documentsWriter.resetDeleteQueue(newQueue);
```

These actions aren't synchronized into a single atomic operation. Meaning, between `newQueue` being created, and calling `getMaxSeqNo`, other code could have executed incrementing the sequence number in the `documentsWriter` class. I found the bug!

![If only it were that easy.](/assets/concurrency-bugs-lucene-debugging/false-hope.jpg)

But, as with most complex bugs, finding the root cause wasn't simple. That's when a hero stepped in.

## A hero in the fray

Enter our hero: [Ao Li](https://aoli.al/) and his colleagues at the PASTA Lab. I will let him explain how they saved the day with Fray.

[Fray](https://github.com/cmu-pasta/fray) is a deterministic concurrency testing framework developed by researchers at the [PASTA Lab](https://pastalab.org/), Carnegie Mellon University. The motivation behind building Fray stems from a noticeable gap between academia and industry: while deterministic concurrency testing has been extensively studied in academic research for over 20 years, practitioners continue to rely on stress testing—a method widely acknowledged as unreliable and flaky—to test their concurrent programs. Thus, we wanted to design and implement a deterministic concurrency testing framework with generality and practical applicability as the primary goal.

## The core idea

At its heart, Fray leverages a straightforward yet powerful principle: sequential execution. Java's concurrency model provides a key [property](https://docs.oracle.com/javase/specs/jls/se8/html/jls-17.html#jls-17.4.3)—if a program is free of data races, all executions will appear sequentially consistent. This means the program's behavior can be represented as a sequence of program statements.

Fray operates by running the target program in a sequential manner: at each step, it pauses all threads except one, allowing Fray to precisely control thread scheduling. Threads are selected randomly to simulate concurrency, but the choices are recorded for subsequent deterministic replay. To optimize execution, Fray only performs context-switches when a thread is about to execute a synchronizing instruction such as locking or atomic/volatile access. A nice property about data-race freedom is that this limited context switching is sufficient to explore all observable behaviors due to any thread interleaving ([our paper](https://arxiv.org/abs/2501.12618) has a proof sketch).

## The challenge: controlling thread scheduling

While the core idea seems simple, implementing Fray presented significant challenges. To control thread scheduling, Fray must manage the execution of each application thread. At first glance, this might seem straightforward—replacing concurrency primitives with customized implementations. However, concurrency control in the JVM is intricate, involving a mix of [bytecode instructions](https://docs.oracle.com/javase/specs/jvms/se21/html/jvms-6.html), [high-level libraries](https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/util/concurrent/locks/ReentrantLock.html), and [native methods](https://github.com/openjdk/jdk/blob/b720517cb33c2119ec6ed85504bce321de748228/src/java.base/share/classes/java/lang/Object.java#L394).

This turned out to be a rabbit hole:

- For example, every `MONITORENTER` instruction must have a corresponding `MONITOREXIT` in the same method. If Fray replaces `MONITORENTER` with a method call to a stub/mock, it also needs to replace `MONITOREXIT`.
- In code that makes use of `object.wait/notify`, if `MONITORENTER` is replaced, the corresponding `object.wait` must also be replaced. This replacement chain extends to `object.notify` and beyond.
- JVM invokes certain concurrency-related methods (e.g., `object.notify` when a thread ends) within native code. Replacing these operations would require modifying the JVM itself.
- JVM functions, such as class loaders and garbage collection (GC) threads, also use concurrency primitives. Modifying these primitives can create mismatches with those JVM functions.
- Replacing concurrency primitives in the JDK often results in JVM crashes during its initialization phase.

These challenges made it clear that a comprehensive replacement of concurrency primitives was not feasible.

## Our solution: shadow lock design

To address these challenges, Fray uses a novel shadow lock mechanism to orchestrate thread execution without replacing concurrency primitives. Shadow locks act as intermediaries that guide thread execution. For example, before acquiring a lock, an application thread must interact with its corresponding shadow lock. The shadow lock determines whether the thread can acquire the lock. If the thread cannot proceed, the shadow lock blocks it and allows other threads to execute, avoiding deadlocks and allowing controlled concurrency. This design enables Fray to control thread interleaving transparently while preserving the correctness of concurrency semantics. Each concurrency primitive is carefully modeled within the shadow lock framework to ensure soundness and completeness. More technical details can be found in our paper.

Moreover, this design is intended to be future-proof. By requiring only the instrumentation of shadow locks around concurrency primitives, it ensures compatibility with newer versions of JVM. This is feasible because the interfaces of concurrency primitives in the JVM are relatively stable and have remained unchanged for years.

## Testing Fray

After building Fray, the next step was evaluation. Fortunately, many applications, such as Apache Lucene, already include concurrency tests. Such concurrency tests are regular JUnit tests that spawn multiple threads, do some work, then (usually) wait for those threads to finish, and then assert some property. Most of the time, these tests pass because they exercise only one interleaving. Worse yet, some tests only fail occasionally in the CI/CD environment, as described earlier, making these failures extremely difficult to debug. When we executed the same tests with Fray, we uncovered numerous bugs. Notably, Fray rediscovered previously reported bugs that had remained unfixed due to the lack of a reliable reproduction, including this blog's focus: [`TestIDVersionPostingsFormat.testGlobalVersions`](https://github.com/apache/lucene/issues/13127). Luckily, with Fray, we can deterministically replay them and provide developers with detailed information, enabling them to reliably reproduce and fix the issue.

## Next steps for Fray

We are thrilled to hear from developers at Elastic that Fray has been helpful in debugging concurrency bugs. We will continue to work on Fray to make it available to more developers.

Our short-term goals include enhancing Fray's ability to deterministically replay the schedule, even in the presence of other non-deterministic operations such as a random-value generator or the use of `object.hashcode`. We also aim to improve the usability of Fray, enabling developers to analyze and debug existing concurrency tests without any manual intervention. Most importantly, if you are facing challenges debugging or testing concurrency issues in your program, we'd love to hear from you. Please don't hesitate to create an issue in the [Fray Github repository](https://github.com/cmu-pasta/fray).

## Time to fix the concurrency bug

Thanks to Ao Li and the PASTA lab, we now have a reliably failing instance of this test! We can finally fix this thing. The key issue resided in how [`DocumentsWriterPerThreadPool`](https://github.com/apache/lucene/blob/f2e7ae40af0b28b1d5f2edc31f8858229a8523f4/lucene/core/src/java/org/apache/lucene/index/DocumentsWriterPerThreadPool.java) allowed for thread and resource reuse.

```
1> new Writer: DocumentsWriterPerThread [pendingDeletes=gen=0, segment=_0, aborted=false, numDocsInRAM=0, deleteQueue=DWDQ: [ generation: 0 ], 0 deleted docIds]
1> DWDQ: [ generation: 0 ] id: 245403753 tid: t0 20 getNextSequenceNumber 1 called from stack:
<snip>...</snip>
1> new Writer: DocumentsWriterPerThread [pendingDeletes=gen=0, segment=_1, aborted=false, numDocsInRAM=0, deleteQueue=DWDQ: [ generation: 0 ], 0 deleted docIds]
1> new Writer: DocumentsWriterPerThread [pendingDeletes=gen=0, segment=_5, aborted=false, numDocsInRAM=0, deleteQueue=DWDQ: [ generation: 0 ], 0 deleted docIds]
1> new Writer: DocumentsWriterPerThread [pendingDeletes=gen=0, segment=_2, aborted=false, numDocsInRAM=0, deleteQueue=DWDQ: [ generation: 0 ], 0 deleted docIds]
1> new Writer: DocumentsWriterPerThread [pendingDeletes=gen=0, segment=_6, aborted=false, numDocsInRAM=0, deleteQueue=DWDQ: [ generation: 0 ], 0 deleted docIds]
1> new Writer: DocumentsWriterPerThread [pendingDeletes=gen=0, segment=_3, aborted=false, numDocsInRAM=0, deleteQueue=DWDQ: [ generation: 0 ], 0 deleted docIds]
1> new Writer: DocumentsWriterPerThread [pendingDeletes=gen=0, segment=_4, aborted=false, numDocsInRAM=0, deleteQueue=DWDQ: [ generation: 0 ], 0 deleted docIds]
```

Here we can see each thread being created, referencing the initial delete queue at generation 0.

Then the queue advance will occur on flush, correctly seeing the previous 7 actions in the queue.

```
1> DWDQ: [ generation: 0 ] id: 245403753 tid: t020 advanceQueue called from stack with maxSeq 9 lastSeqNo: 1 maxNumPendingOps: 7:
<snip>...</snip>
1> DWDQ: [ generation: 0 ] id: 245403753 tid: t525 getNextSequenceNumber 2 called from stack:
1> DWDQ: [ generation: 0 ] id: 245403753 tid: t828 getNextSequenceNumber 3 called from stack:
1> DWDQ: [ generation: 0 ] id: 245403753 tid: t727 getNextSequenceNumber 4 called from stack:
1> DWDQ: [ generation: 0 ] id: 245403753 tid: t626 getNextSequenceNumber 5 called from stack:
1> DWDQ: [ generation: 0 ] id: 245403753 tid: t424 getNextSequenceNumber 6 called from stack:
1> DWDQ: [ generation: 0 ] id: 245403753 tid: t323 getNextSequenceNumber 7 called from stack:
```

But, before all the threads can finish flushing, two are reused for an additional document:

```
1> getAndLock: DocumentsWriterPerThread [pendingDeletes=gen=0, segment=_0, aborted=false, numDocsInRAM=1, deleteQueue=DWDQ: [ generation: 0 ], 0 deleted docIds]
1> getAndLock: DocumentsWriterPerThread [pendingDeletes=gen=0, segment=_3, aborted=false, numDocsInRAM=1, deleteQueue=DWDQ: [ generation: 0 ], 0 deleted docIds]
```

These will then increment the `seqNo` above the assumed maximum, which was calculated during the flush as 7. Note the additional `numDocsInRAM` for segments `_3` and `_0`.

```
1> DocumentsWriterPerThread [pendingDeletes=gen=0, segment=_2, aborted=false, numDocsInRAM=1, deleteQueue=DWDQ: [ generation: 0 ], 0 deleted docIds] checkout to remove
1> DocumentsWriterPerThread [pendingDeletes=gen=0, segment=_3, aborted=false, numDocsInRAM=2, deleteQueue=DWDQ: [ generation: 0 ], 0 deleted docIds] checkout to remove
1> DocumentsWriterPerThread [pendingDeletes=gen=0, segment=_6, aborted=false, numDocsInRAM=1, deleteQueue=DWDQ: [ generation: 0 ], 0 deleted docIds] checkout to remove
1> DocumentsWriterPerThread [pendingDeletes=gen=0, segment=_4, aborted=false, numDocsInRAM=1, deleteQueue=DWDQ: [ generation: 0 ], 0 deleted docIds] checkout to remove
1> DocumentsWriterPerThread [pendingDeletes=gen=0, segment=_5, aborted=false, numDocsInRAM=1, deleteQueue=DWDQ: [ generation: 0 ], 0 deleted docIds] checkout to remove
1> DocumentsWriterPerThread [pendingDeletes=gen=0, segment=_1, aborted=false, numDocsInRAM=1, deleteQueue=DWDQ: [ generation: 0 ], 0 deleted docIds] checkout to remove
1> DocumentsWriterPerThread [pendingDeletes=gen=0, segment=_0, aborted=false, numDocsInRAM=2, deleteQueue=DWDQ: [ generation: 0 ], 0 deleted docIds] checkout to remove
```

Thus causing Lucene to incorrectly account for the sequence of document actions during a flush and tripping this test failure.

Like all good bug fixes, the actual fix is about [10 lines of code](https://github.com/apache/lucene/pull/13627/files). But took two engineers multiple days to actually figure out:

![Some lines of code take longer to write than others. And even require the help of some new friends.](/assets/concurrency-bugs-lucene-debugging/fix-lines-of-code.jpg)

## Not all heroes wear capes

Yes, it's cliche – but it's true.

Concurrent program debugging is incredibly important. These tricky concurrency bugs take an inordinate amount of time to debug and work through. While new languages like Rust have built in mechanisms to help prevent race conditions like this, the majority of software in the world is already written, and written in something other than [Rust](https://www.rust-lang.org/). Java, even after all these years, is still one of the most used languages. Improving debugging on JVM based languages makes the software engineering world better. And given how some folks think that code will be written by Large Language Models, maybe our jobs as engineers will eventually just be debugging bad LLM code instead of just our own bad code. But, no matter the future of software engineering, concurrent program debugging will remain critical for maintaining and building software.

Thank you Ao Li and his colleagues from the PASTA Lab for making it that much better.
