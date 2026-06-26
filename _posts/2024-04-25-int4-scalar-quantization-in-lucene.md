---
layout: post
title: "Understanding Int4 scalar quantization in Lucene"
published: true
use_math: true
tags: [elastic, elasticsearch, lucene, vector, quantization]
---

> Originally published on [Elasticsearch Labs](https://www.elastic.co/search-labs/blog/int4-scalar-quantization-in-lucene).

## How does `Int4` quantization work in Lucene

### Storing and scoring the quantized vectors

Lucene stores all the vectors in a flat file, making it possible for each vector to be retrieved given some ordinal. You can read a brief overview of this in our [previous scalar quantization blog](https://www.elastic.co/search-labs/blog/scalar-quantization-in-lucene#quantization-per-segment).

Now `int4` gives us additional compression options than what we had before. It reduces the quantization space to only 16 possible values (0 through 15). For more compact storage, Lucene uses some simple bit shift operations to pack these smaller values into a single byte, allowing a possible 2x space savings on top of the already 4x space savings with int8. In all, storing int4 with bit compression is 8x smaller than `float32`.

![int4 byte compression](/assets/int4-scalar-quantization-in-lucene/int4-byte-compression.png)
*Figure 1: This shows the reduction in bytes required with `int4` which allows an 8x reduction in size from `float32` when compressed.*

`int4` also has some benefits when it comes to scoring latency. Since the values are known to be between `0-15`, we can take advantage of knowing exactly when to worry about value overflow and optimize the dot-product calculation. The maximum value for a dot product is `15*15=225` which can fit in a single byte. ARM processors (like my macbook) have a [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) instruction length of 128 bits (16 bytes). This means that for a Java `short` we can allocate 8 values to fill the lanes. For 1024 dimensions, each lane will end up accumulating a total of `1024/8=128` multiplications that have a max value of `225`. The resulting maximum sum of `28800` fits well within the limit of Java's `short` value and we can iterate more values at a time than. Here is some simplified code of what this looks like for ARM.

```java
// snip preamble handling vectors longer than 1024
// 8 lanes of 2 bytes 
ShortVector acc = ShortVector.zero(ShortVector.SPECIES_128);
for (int i = 0; i < length; i += ByteVector.SPECIES_64.length()) {
  // Get 8 bytes from vector a
  ByteVector va8 = ByteVector.fromArray(ByteVector.SPECIES_64, a, i);
  // Get 8 bytes from vector b
  ByteVector vb8 = ByteVector.fromArray(ByteVector.SPECIES_64, b, i);
  // Multiply together, potentially saturating signed byte with a max of 225
  ByteVector prod8 = va8.mul(vb8);
  // Now convert the product to accumulate into the short
  ShortVector prod16 = prod8.convertShape(B2S, ShortVector.SPECIES_128, 0).reinterpretAsShorts();
  // Ensure to handle potential byte saturation
  acc = acc.add(prod16.and((short) 0xFF));
}
// snip, tail handling
```

### Calculating the quantization error correction

For a more detailed explanation of the error correction calculation and its derivation, please see [error correcting the scalar dot-product](https://www.elastic.co/search-labs/blog/vector-db-optimized-scalar-quantization#error-correcting-the-scalar-dot-product).

Here is a short summary, woefully (or joyfully) devoid of complicated mathematics.

For every quantized vector stored, we additionally keep track of a quantization error correction. Back in the [Scalar Quantization 101 blog](https://www.elastic.co/search-labs/blog/scalar-quantization-101#time-to-remember-your-algebra) there was a particular constant mentioned:

$
\alpha \times int8_i \times min
$

This constant is a simple constant derived from basic algebra. However, we now include additional information in the stored float that relates to the rounding loss.

$
\sum_{i=0}^{dim-1} ((i - min) - i'\times\alpha)i'\times\alpha
$

Where $i$ is each floating point vector dimension, $i'$ is the scalar quantized floating point value, and $\alpha=\frac{max - min}{(1 \ll bits) - 1}$.

This has two consequences. The first is intuitive, as it means that for a given set of quantization buckets, we are slightly more accurate as we account for some of the lossiness of the quantization. The second consequence is a bit more nuanced. It now means we have an error correction measure that is impacted by the quantization bucketing. This implies that it can be optimized.

### Finding the optimal bucketing for int4 quantization

The naive and simple way to do scalar quantization can get you pretty far. Usually, you pick a confidence interval from which you calculate the allowed extreme boundaries for vector values. The default in Lucene and consequently Elasticsearch is $1-1/(dimensions+1)$. Figure 2 shows the confidence interval over some sampled [CohereV3 embeddings](https://huggingface.co/datasets/Cohere/wikipedia-2023-11-embed-multilingual-v3/viewer/en). Figure 3 shows the same vectors, but scalar quantized with that statically set confidence interval.

![CohereV3 vector dimension distribution](/assets/int4-scalar-quantization-in-lucene/cohere-v3-distribution.png)
*Figure 2: A sampling of CohereV3 dimension values.*

![CohereV3 vector dimension distribution quantized](/assets/int4-scalar-quantization-in-lucene/cohere-v3-distribution-quantized.png)
*Figure 3: CohereV3 dimension values quantized into int7 values. What are those spikes at the end? Well, that is the result of truncating extreme values during the quantization process.*

But, we are leaving some nice optimizations on the floor. What if we could tweak the confidence interval to shift the buckets, allowing for more important dimensional values to have higher fidelity. To optimize, Lucene does the following:

- Sample around 1,000 vectors from the data set and calculate their true nearest 10 neighbors.
- Calculate a set of candidate upper and lower quantiles. The set is calculated by using two different confidence intervals: $1 - 1/(dimensions+1)$ and $1-(dimensions/10)/(dimensions + 1)$. These intervals are on the opposite extremes. For example, vectors with `1024` dimensions would search quantile candidates between confidence intervals `0.99902` and `0.90009`.
- Do a grid search over a subset of the quantiles that exist between these two confidence intervals. The grid search finds the quantiles that maximize the [coefficient of determination](https://en.wikipedia.org/wiki/Coefficient_of_determination) of the quantization score errors vs. the true 10 nearest neighbors calculated earlier.

![Searching the int4 space](/assets/int4-scalar-quantization-in-lucene/int4-space-search.gif)
*Figure 3: Lucene searching the confidence interval space and testing various buckets for `int4` quantization.*

![The optimal int4 quantization](/assets/int4-scalar-quantization-in-lucene/int4-optimal-quantization.png)
*Figure 4: The best `int4` quantization buckets found for this CohereV3 sample set.*

For a more complete explanation of the optimization process and the mathematics behind this optimization, see [optimizing the truncation interval](https://www.elastic.co/search-labs/blog/vector-db-optimized-scalar-quantization#optimizing-the-truncation-interval).

### Speed vs. size for quantization

As I mentioned before, `int4` gives you an interesting tradeoff between performance and space. To drive this point home, here are some memory requirements for CohereV3 500k vectors.

![CohereV3 Memory Requirements](/assets/int4-scalar-quantization-in-lucene/cohere-v3-memory-requirements.png)
*Figure 5: Memory requirements for CohereV3 500k vectors.*

Of course, we see the typical 4x reduction in regular scalar quantization, but then additional 2x reduction with `int4`. Moving the required memory from `2GB` to less than `300MB`. Keep in mind, this is with compression enabled. Decompressing and compressing bytes does have an overhead at search time. For every byte vector, we must decompress them before doing the `int4` comparisons. Consequently, when this is introduced in Elasticsearch, we want to give users the ability to choose to compress or not. For some users, the cheaper memory requirements are just too good to pass up, for others, their focus is speed. `Int4` gives the opportunity to tune your settings to fit your use-case.

![CohereV3 Speed](/assets/int4-scalar-quantization-in-lucene/cohere-v3-speed.png)
*Figure 6: HNSW graph search speed comparison for CohereV3 500k vectors.*

### Speed part 2: more SIMD in int4

Figure 6 is a bit disappointing in terms of the speed of compressed scalar quantization. We expect performance benefits from loading fewer bytes to the JVM heap. However, this is being outweighed by the cost of decompressing them. This caused us dig deeper. The reason for the performance impact was naively decompressing the bytes separately from the dot-product comparison. This is a mistake. We can do better.

Consequently, we can use SIMD to decompress the bytes and compare them in the same function. This is a bit more complicated than the previous SIMD example, but it is possible. Here is a simplified version of what this looks like for ARM.

```java
// the packed vector, each byte contains two values
// for packed value `n`: packed[n] = (raw[n] << 4) | raw[packed.length + n];
ByteVector vb8 = ByteVector.fromArray(ByteVector.SPECIES_64, packed, i + j);
// unpacked, the raw query vector int4 quantized
ByteVector va8 = ByteVector.fromArray(ByteVector.SPECIES_64, unpacked, i + j + packed.length);

// upper side, decompress and multiply
ByteVector prod8 = vb8.and((byte) 0x0F).mul(va8);
ShortVector prod16 = prod8.convertShape(B2S, ShortVector.SPECIES_128, 0).reinterpretAsShorts();
// Ensure to handle potential byte saturation
acc0 = acc0.add(prod16.and((short) 0xFF));

// lower side, decompress and multiply
va8 = ByteVector.fromArray(ByteVector.SPECIES_64, unpacked, i + j);
prod8 = vb8.lanewise(LSHR, 4).mul(va8);
prod16 = prod8.convertShape(B2S, ShortVector.SPECIES_128, 0).reinterpretAsShorts();
// Ensure to handle potential byte saturation
acc1 = acc1.add(prod16.and((short) 0xFF));
```

As expected, this has a significant improvement on ARM. Effectively removing all performance discrepancies on ARM between compressed and uncompressed scalar quantization.

![CohereV3 Speed Improved](/assets/int4-scalar-quantization-in-lucene/cohere-v3-speed-improved.png)
*Figure 7: HNSW graph search comparison with int4 quantized vectors over 500k Coherev3 vectors. This is on ARM architecture.*
