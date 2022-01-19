Two strategies for training using normalized features
* Modification of softmax loss which optimizes cosine similarity instead of inner-product
* Reformulation of metric learning by introducing an agent vector for each class

Some question in practice
* Why is feature normalization so effcient when comparing the CNN features trained by classification loss?
* Why does directly optimizing the cosine similarity using softmax loss cause the network to fail to converge?
* How to optimize a cosine similarity when using softmax loss?
* Since models with softmax loss fail to converge after normalization, are there any other loss functions suitable for normalized features?

Normalization -> accelerate convergence speed

---

The reason why the softmax loss tends to create a radial feature distribution is that the softmax loss actually acts as the soft version of max operator. Scaling the feature vectors' magnitude does not affect the assignment of its class.

$$L_S=-\frac{1}{m}\sum_{i=1}^m\log\frac{e^{W_{y_i}^Tf_i+b_{y_i}}}{\sum_{j=1}^ne^{W_j^Tf_i+b_j}}$$

* $m$ is the number of training samples
* $n$ is the number of classes
* $f_i$ is the feature of the i-th sample
* $y_i$ is the corresponding label in range $[1,n]$
* $W$ and $b$ are the weight matrix and the bias vector of the last inner-product layer before the softmax loss

In testing phase, we classify a sample by:

$$\text{Class}(f)=\arg\max_i(W_i^Tf+b_i)$$

TODO: proof proposition 1

This proposition implies that softmax loss always encourages well-seperated features to have bigger magnitudes. This is the reason why the feature distribution is "radial". By normalization, we can eliminate its effect. Thus, we usually use the cosine of two feature vectors to measure the similarity of two samples.

?If a bias term is added after the inner-product operation, then normalization may cause feature clusters to overlap each other.

## Layer Definition
To prevent dividing zero, we add a small positive value $\varepsilon$ to original 2-norm, thus $L_2$ normalization is redefined as follows:

$$\bar{x}=\frac{x}{\|x\|_2}=\frac{x}{\sqrt{\sum_ix_i^2+\varepsilon}}$$

TODO: induce chain rule

&emsp;&emsp;**lemma 1**: The derivative of vector $\pmb{x}$ is orthogonal to $\pmb{x}$ if and only if $\pmb{x}$ has constant magnitude. (Dot product gives how much the two vectors are pointing in the same direction)

$$\frac{\partial\sum_ix_i^2}{\partial t}=0\iff\pmb{x}\cdot\frac{\partial\pmb{x}}{\partial t}=0$$

Thus $x$ and $\partial L/\partial x$ are orthogonal to each other. TODO: complete the proof