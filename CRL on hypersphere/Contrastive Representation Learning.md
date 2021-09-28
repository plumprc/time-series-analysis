---
title: 《Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere》阅读笔记
date: 2021-09-23 14:53:21
categories:
- 机器学习
tags:
- 机器学习
- 表示学习
- 自监督学习
---

<center>论文地址：<a href="https://arxiv.org/abs/2005.10242">Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere</a></center>

## Background and Motivation
Two key properties related to the contrastive loss
* alignment (closeness) of features from positive pairs
* uniformity of the induced distribution of the (normalized) features on the hypersphere

Unit $L_2$ norm constraint effectively restricts
the output space to the unit hypersphere.

TODO: necessity of normalization

Intuitively, having the features live on the unit hypersphere leads to several desirable traits:
* Fixed-norm vectors are known to improve training stability (avoid degradient? and benefit to computation)
* Well clustered features are linearly separable with the rest of feature space

While the unit hypersphere is a popular choice of feature space, not all encoders that map onto it are created equal. Recent works argue that representations should additionally be invariant to unnecessary details, and preserve as much information as possible.
* Alignment favors encoders that assign similar features to similar samples
* Uniformity prefers a feature distribution that preserves maximal information

Contrastive representation learning in fact directly optimizes these two properties in the limit of infinite negative samples.

---

Current unsupervised contrastive representation learning approachs are almost motivated by InfoMax principle, maximizing $I(f(x),f(y))$ for $(x,y)\sim p_\text{pos}$, which is usually explained as a lower bound to contrastive loss. However, this interpretation is known to be inconsistent with the actual behavior in practice. (e.g. optimizing a tighter bound on MI can lead to worse representations)

Latent classes?: large number of negative samples is essential or not?

Unit hypersphere is indeed a nice feature space

Uniformly distributing points on the unit hypersphere is a well-studied problem.
* Minimizing the total pairwise potential

---

&emsp;&emsp;Let $p_\text{data}$ be the data distribution over $\mathbb{R}^n$ and $p_\text{pos}$ the distribution of positive pairs over $\mathbb{R}^n\times\mathbb{R}^n$. Based on empirical practices, we assume that:
* Symmetry: $\forall x,y,p_\text{pos}(x,y)=p_\text{pos}(y,x)$
* ?Matching marginal: $\forall x,\int p_\text{pos}(x,y)\rm dy=p_\text{data}(x)$

Question: how to balance the contribution of positive samples and negative samples? I do not think that we can get the distribution just from positive samples.

?why m-1: polar coordinates with $r=1$

$$L_\text{contrastive}(f;\tau,M)=\mathbb{E}-\log\frac{e^{f(x)^Tf(y)/\tau}}{e^{f(x)^Tf(y)/\tau}+\sum_ie^{f(x_i^-)^Tf(y)/\tau}}$$

* $f$ is a kernel function mapping original data distribution to latent space
* $\tau>0$ is a scalar temperature hyperparameter
* $M\in\mathbb{Z}^+$ is a fixed number of negative samples

### Necessity of normalization
Without the norm constraint, the `softmax` distribution can be made arbitrarily sharp by simply scaling all the features.

### Feature Distribution on the Hypersphere
Contrastive loss should prefer two following properties:
* Alignment: positive pairs should be mapped to nearby feature, and thus be mostly invariant to unneeded noise factors
* Uniformity: feature vectors should be roughly uniformly distributed on the unit hypersphere $S^{m-1}$, preserving as much information of the data as possible

Suppose the encoder is perfectly aligned ($f(x)=f(y)$), in a unit hypersphere sapce we can get $f(x)^Tf(y)=1$. Then minimizing the loss is equivalent to optimizing:

$$\mathbb{E}\log(e^{1/\tau}+\sum_ie^{f(x_i^-)^Tf(x)/\tau})$$

which is akin to maximizing pairwise distances with a `LogSumExp` transformation. Intuitively, pushing all features away from each other should indeed cause them to be roughly uniformly distributed.

### Quantifying Alignment and Uniformity
&emsp;&emsp;The alignment loss is directly defined with the expected distance between positive pairs:

$$L_\text{align}(f;\alpha)=\mathbb{E}\|f(x)-f(y)\|_2^\alpha,\quad\alpha>0$$

&emsp;&emsp;We desire that optimizing uniformity can converge original distribution to uniform distribution.  To this end, we consider the Gaussian potential kernel $G_t:S^d\times S^d\rightarrow\mathbb{R}^+$.

$$G_t(u,v)=e^{-t\|u-v\|_2^2},\quad t>0$$

Thus the uniformity loss is defined as the logarithm of the average pairwise Gaussian potential:

$$L_\text{uniform}(f;t)=\log\mathbb{E}e^{-t\|f(x)-f(y)\|_2^2}$$

TODO: see appendix to proof that Gaussian potential kernel is useful

---

&emsp;&emsp;**theorem 1**: For fixed $\tau>0$, as the number of negative samples $M\rightarrow\infty$, the normalized contrastive loss converges to

$$\lim_{M\rightarrow\infty}L_\text{contrastive}(f;\tau,M)-\log M=$$

TODO: appendix proof

---

## Proof and Proof and Proof
...

## Summary
Main contributions are:
* ...
