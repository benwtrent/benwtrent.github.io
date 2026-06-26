---
layout: post
title: "Scalar quantization 101"
published: true
use_math: true
tags: [elastic, elasticsearch, lucene]
---

> Originally published on [Elasticsearch Labs](https://www.elastic.co/search-labs/blog/scalar-quantization-101).

## Introduction to scalar quantization

Most embedding models output $float32$ vector values. While this provides the highest fidelity, it is wasteful given the information that is actually important in the vector. Within a given data set, embeddings never require all 2 billion options for each individual dimension. This is especially true on higher dimensional vectors (e.g. 386 dimensions and higher). Quantization allows for vectors to be encoded in a lossy manner, thus reducing fidelity slightly with huge space savings.

## Understanding buckets in scalar quantization

Scalar quantization takes each vector dimension and buckets them into some smaller data type. For the rest of the blog, we will assume quantizing $float32$ values into $int8$. To bucket values accurately, it isn't as simple as rounding the floating point values to the nearest integer. Many models output vectors that have dimensions continuously on the range $[-1.0, 1.0]$. So, two different vector values 0.123 and 0.321 could both be rounded down to 0. Ultimately, a vector would only use 2 of its 255 available buckets in $int8$, losing too much information.

![Quantization illustration](/assets/scalar-quantization-101/quantization-illustration.png)
*Figure 1: Illustration of quantization goals, bucketing continuous values from $-1.0$ to $1.0$ into discrete $int8$ values.*

The math behind the numerical transformation isn't too complicated. Since we can calculate the minimum and maximum values for the floating point range, we can use [min-max normalization](https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)) and then linearly shift the values.

$
int8 \approx  \frac{127}{max - min} \times (float32 - min)
$

$
float32 \approx \frac{max - min}{127} \times int8 + min
$

*Figure 2: Equations for transforming between $int8$ and $float32$. Note, these are lossy transformations and not exact. In the following examples, we are only using positive values within int8. This aligns with the Lucene implementation.*

## The role of statistics in scalar quantization

A [quantile](https://en.wikipedia.org/wiki/Quantile) is a slice of a distribution that contains a certain percentage of the values. So, for example, it may be that $99\%$ of our floating point values are between $[-0.75, 0.86]$ instead of the true minimum and maximum values of $[-1.0, 1.0]$. Any values less than -0.75 and greater than 0.86 are considered outliers. If you include outliers when attempting to quantize results, you will have fewer available buckets for your most common values. And fewer buckets can mean less accuracy and thus greater loss of information.

![Quantile illustration](/assets/scalar-quantization-101/quantile-illustration.png)
*Figure 3: Illustration of the $99\%$ [confidence interval](https://en.wikipedia.org/wiki/Confidence_interval) and the individual quantile values. $99\%$ of all values fall within the range $[-0.75, 0.86]$.*

This is all well and good, but now that we know how to quantize values, how can we actually calculate distances between two quantized vectors? Is it as simple as a regular [dot_product](https://en.wikipedia.org/wiki/Dot_product)?

## The role of algebra in scalar quantization

We are still missing one vital piece, how do we calculate the distance between two quantized vectors. While we haven't shied away from math yet in this blog, we are about to do a bunch more. Time to break out your pencils and try to remember [polynomials](https://en.wikipedia.org/wiki/Polynomial) and basic algebra.

The basic requirement for [dot_product](https://en.wikipedia.org/wiki/Dot_product) and [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) is being able to multiply floating point values together and sum up their results. We already know how to transform between $float32$ and $int8$ values, so what does multiplication look like with our transformations?

$
float32_i \times float32'_i \approx (\frac{max - min}{127} \times int8_i + min) \times (\frac{max - min}{127} \times int8'_i + min)
$

We can then expand this multiplication and to simplify we will substitute $\alpha$ for $\frac{max - min}{127}$.

$
\alpha^2 \times int8_i \times int8'_i + \alpha \times int8_i \times min + \alpha \times int8'_i \times min + min^2
$

What makes this even more interesting, is that only one part of this equation requires both values at the same time. However, dot_product isn't just two floats being multiplied, but all the floats for each dimension of the vector. With vector dimension count $dim$ in hand, all the following can be pre-calculated at query time and storage time.

$dim\times\alpha^2$ is just $dim\times(\frac{max-min}{127})^2$ and can be stored as a single float value.

$
\sum_{i=0}^{dim-1}min\times\alpha\times int8_i
$

and

$
\sum_{i=0}^{dim-1}min\times\alpha\times int8'_i
$

can be pre-calculated and stored as a single float value or calculated once at query time.

$
dim\times min^2
$ 

can be pre-calculated and stored as a single float value.

Of all this:

$$
dim \times \alpha^2 \times dotProduct(int8, int8') + \sum_{i=0}^{dim-1}min\times\alpha\times int8_i + \sum_{i=0}^{dim-1}min\times\alpha\times int8'_i + dim\times min^2
$$

The only calculation required for dot_product is just $dotProduct(int8, int8')$ with some pre-calculated values combined with the result.

## Ensuring accuracy in quantization

So, how is this accurate at all? Aren't we losing information by quantizing? Yes, we are, but quantization takes advantage of the fact that we don't need all the information. For learned embeddings models, the distributions of the various dimensions usually don't have [fat-tails](https://en.wikipedia.org/wiki/Fat-tailed_distribution). This means they are localized and fairly consistent. Additionaly, the error introduced per dimension via quantization is independent. Meaning, the error cancels out for our typical vector operations like dot_product.
