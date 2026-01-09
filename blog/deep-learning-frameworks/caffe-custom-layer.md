---
layout: post
title: "Caffe Custom Sigmoid Cross Entropy Loss Layer"
permalink: /blog/deep-learning-frameworks/caffe-custom-layer.html
---

**2017-06-15 00:00** | *Caffe*

[← Back to Blog](../index.html)

---

# Custom sigmoid cross entropy loss caffe layer
Here, we implement a custom sigmoid cross entropy loss layer for caffe.

A modification of this layer was used for U-net architecture model which can be seen in the image below, the layer being implemented in this post is marked `custom_loss`.

💡 *Tip: For better visualization, right click on the image and click "Open Image in New Tab".*

![U-Net Architecture](../Images/akshay_unet.png)

Sigmoid cross entropy loss may be defined as
$$t\ln(S) + (1-t) \ln(1-S)$$
where `t` is the `target` or `label` and `S` is the `Sigmoid score`. Replace `S` with sigmoid we obtain the following where `x` is the `score`.
$$t\ \ln\Big(\frac{1}{1+e^{-x}}\Big) + (1-t)\ \ln\ \Big(1-\frac{1}{1+e^{-x}}\Big)$$
Multiplying $e^{x}$ to the numerator and denominator terms of the second term.
$$t\ \ln\Big(\frac{1}{1+e^{-x}}\Big) + (1-t)\ \ln\ \Big(\frac{e^{-x}}{1+e^{-x}}\Big)$$
$$t\ \ln\Big(\frac{1}{1+e^{-x}}\Big) + \ln\ \Big(\frac{e^{-x}}{1+e^{-x}}\Big) -t \ln\ \Big(\frac{e^{-x}}{1+e^{-x}}\Big)$$
converting $log\ (a/b)$ to $log\ a - log\ b$
$$t\ \Big[\ln{1}-\ln(1+e^{-x})\Big] + \Big[\ln\ {e^{-x}} - \ln\ ({1+e^{-x}})\Big] -t\ \Big[\ln\ {e^{-x}}-\ln\ ({1+e^{-x}})\Big]$$
$$\Big[-t\ln(1+e^{-x})\Big] + \ln\ {e^{-x}} - {\ln\ ({1+e^{-x}})} -t\ln\ {e^{-x}}\Big[+t\ln\ ({1+e^{-x}})\Big]$$
Cancelling the first and the last bracketed terms as they are of oppositte signs.
$$\ln\ {e^{-x}} - {\ln\ ({1+e^{-x}})} -t\ln\ {e^{-x}}$$
Converting $e^a$ to $ae$
$${-x}\ln\ {e} - {\ln\ ({1+e^{-x}})} + xt \ln\ {e}$$
Using $\ln{e} = 1$
$${-x} - {\ln\ ({1+e^{-x}})} + xt $$
$$xt - x - {\ln\ ({1+e^{-x}})} \tag{1} $$

Now, lets examine the nature of the function $e^{-x}$ and $e^{x}$

<div style="display: flex; justify-content: space-around; align-items: flex-start; gap: 20px; margin: 20px 0;">
  <div style="flex: 1; text-align: center;">
    <h4>e<sup>-x</sup> function</h4>
    <img src="../Images/e-x.png" alt="e^-x function" style="max-width: 100%; height: auto;">
  </div>
  <div style="flex: 1; text-align: center;">
    <h4>e<sup>x</sup> function</h4>
    <img src="../Images/ex.png" alt="e^x function" style="max-width: 100%; height: auto;">
  </div>
</div>

As we can see $e^{-x}$ drops  with incresing `x`, becoming very large for high negative values of `x`. These high values could easily cause an overflow situation, meaning the high values can't be accomodated in the data type being used. To avoid such an overflow situation, we may modify the equation to use $e^x$ when $x<0$.<br>
Hence we convert (1) to obtain a modified version with $e^x$.

$$xt - x - {\ln\ ({1+e^{-x}})}$$

Converting $-log\ a$ to $log\frac{1}{a}$
$$xt - x + {\ln\ \frac{1}{1+e^{-x}}} $$
Multiplying the numerator and denominator of the log term with $e^x$
$$xt - x + {\ln\ \frac{1*e^{x}}{(1+e^{-x})*e^{x}}} $$

$$xt - x + \ln\ \frac{e^{x}}{e^{x}+1} $$

$$xt - x + \Big[\ln\ {e^{x}} -\ln ({e^{x}+1})\Big] $$

$$xt - x + x\ln\ {e} -\ln ({e^{x}+1}) $$

$$xt - x + x -\ln ({e^{x}+1}) $$

$$xt -\ln ({e^{x}+1}) \tag{2} $$

Therefore, we will use the equation $(1)$ for $x>0$ and equation $(2)$ for $x<0$.<br>
Lets attempt to combine the two equations into one. Rewriting both equations:

$$xt - x - {\ln\ ({1+e^{-x}})}  \tag{x>0}$$
$$xt - 0 - {\ln\ ({1+e^{x}})}  \tag{x<0}$$

$$xt - \max(x,0) - {\ln\ ({1+e^{-\left|x\right|}})}  \tag{For all x}$$


Lets find the derative of $(1)$
$$ \frac {\partial}{\partial x} \Big(xt - x - {\ln\ ({1+e^{-x}})}\Big)$$
$$ \frac {\partial xt}{\partial x} - \frac {\partial x}{\partial x} - \frac {\partial}{\partial x}\Big({\ln\ ({1+e^{-x}})}\Big)$$
Applying $\frac {\partial \log {x}}{\partial x}=\frac{1}{x}$ and chain rule
$$ t - 1 - \frac {1}{1+e^{-x}} * \frac {\partial}{\partial x}{({1+e^{-x}})}$$
$$ t - 1 - \frac {1}{1+e^{-x}} * \frac {\partial}{\partial x}{({e^{-x}})}$$
$$ t - 1 + \frac {e^{-x}}{1+e^{-x}}$$
Rearranging above equation
$$ t + \frac {e^{-x}}{1+e^{-x}} - 1$$
$$ t + \frac {e^{-x}-1-e^{-x}}{1+e^{-x}}$$
$$ t - \frac {1}{1+e^{-x}}$$
The above fraction is equal to the `Sigmoid` function `S` used in the begining of the post
$$ t - S$$

Lets find the derative of $(2)$
$$\frac {\partial}{\partial x}\Big(xt -\ln ({e^{x}+1})\Big) $$
$$ \frac {\partial xt}{\partial x} - \frac {\partial}{\partial x}\Big({\ln\ ({1+e^{x}})}\Big)$$
Applying $\frac {\partial \log {x}}{\partial x}=\frac{1}{x}$ and chain rule
$$ t - \frac {1}{1+e^{x}} * \frac {\partial}{\partial x}{({1+e^{x}})}$$
$$ t - \frac {1}{1+e^{x}} * \frac {\partial}{\partial x}{({e^{x}})}$$
$$ t - \frac {e^{x}}{1+e^{x}}$$
Multiplying both numerator and denominator with $e^{-x}$
$$ t - \frac {e^{x}*e^{-x}}{(1+e^{x})(e^{-x})}$$
$$ t - \frac {1}{1+e^{-x}}$$
As for equation $(1)$, the above fraction is equals to the `Sigmoid` function `S`
$$ t - S$$
This proves that the derivative in both cases $x<0$ and $x>0$ is the same, and can be given as the difference of the target value to the sigmoid value.

Now, lets write the custom caffe layer. The custom layers in python are written as a class with the inline four major functions. $Labels \in{0,1}$

```python
import caffe
import scipy
```

```python
class CustomSigmoidCrossEntropyLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check for all inputs
        if len(bottom) != 2:
            raise Exception("Need two inputs (scores and labels) to compute sigmoid crossentropy loss.")

    def reshape(self, bottom, top):
        # check input dimensions match between the scores and labels
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference would be the same shape as any input
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # layer output would be an averaged scalar loss
        top[0].reshape(1)

    def forward(self, bottom, top):
        score=bottom[0].data
        label=bottom[1].data

        first_term=np.maximum(score,0)
        second_term=-1*score*label
        third_term=np.log(1+np.exp(-1*np.absolute(score)))

        top[0].data[...]=np.sum(first_term+second_term+third_term)
        sig=scipy.special.expit(score)
        self.diff=(sig-label)
        if np.isnan(top[0].data):
                exit()

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...]=self.diff
```

##### Adding Layer information to the prototxt
We need to add the custom layer to the caffe prototxt

```
layer {
  type: 'Python'
  name: 'loss'
  top: 'loss_opt'
  bottom: 'score'
  bottom: 'label'
  python_param {
    # the module name -- usually the filename -- that needs to be in $PYTHONPATH
    module: 'loss_layers'
    # the layer name -- the class name in the module
    layer: 'CustomSigmoidCrossEntropyLossLayer'
  }
  include {
        phase: TRAIN
  }
  # set loss weight so Caffe knows this is a loss layer.
  # since PythonLayer inherits directly from Layer, this isn't automatically
  # known to Caffe
  loss_weight: 1
}
```


---

[← Back to Blog](../index.html)
