
# 1. Create a Neuron
> 1_Neuron.py

## 1.1 Statement

$a = g(w_1x_1 + w_2x_2 + b)$

$weights =
\begin{bmatrix}
w_1 \\
w_2 \\
\end{bmatrix}
$

$x =
\begin{bmatrix}
x_1 \\
x_2 \\
\end{bmatrix}
$

## 1.2 Dot product
- A.dot(B)
- A.vdot(B)

$
\begin{bmatrix}
w_1 \\
w_2 \\
\end{bmatrix} ·
\begin{bmatrix}
x_1 \\
x_2 \\
\end{bmatrix} = 
w_1x_1 + w_2x_2
$

```python
z = np.vdot(self.weights, x) + self.bias
```

## 1.3 Matrix multiplication
$
\begin{bmatrix}
w_1, w_2
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2 \\
\end{bmatrix} = 
w_1x_1 + w_2x_2
$

= 
($
\begin{bmatrix}
w_1 \\
w_2 \\
\end{bmatrix})^T
\begin{bmatrix}
x_1 \\
x_2 \\
\end{bmatrix} = 
w_1x_1 + w_2x_2
$

```python
W, X = np.matrix(self.weights), np.matrix(x)

z = np.transpose(W) * X + self.bias
```


# 2. Create a simple network
> 2_NN.py

**forward propagation**

```python
"""
1. input layer: x1, x2
2. hidden layer: h1, h2
3. output layer: o1

all weights = [0 1] w1 = 0 w2 = 1
all bias = 0

statements:
h1 = w1x1 + w2x2 + b
"""
```

# 3. Cost Function
- squared error
- mean()

Cost Function = ($(y_1^{predict} - y_1^{train}) ^ 2$ $+ ... + (y_n^{predict} - y_n^{train}) ^ 2$) $/ n$

= $((Y^{predict} - Y^{train}) ^ 2).mean()$

公式:

$\displaystyle\sum_{i=1}^n(Y^{(predict)} - Y^{({train})}) ^ 2 / n$

# 4. Train a Neural Network
Goal: minimize the loss of NN

## 4.1 Math Equation

**(1) Loss as a multivariable function**
$L(w_1, w_2, ..., b_1, b_2, ...)$

**(2) partial derivatives**
![image](https://pic1.zhimg.com/80/v2-ac5aa2b1340c9812674129b6975a7d6c_hd.jpg)

L: loss function (cost function)

**if we optimize w1, how dL will be changed?**

$dL / dw_1 = \frac{dL}{do_1} * \frac{do_1}{dw_1}$ = $\frac{dL}{do_1} * \frac{do_1}{dh_1} * \frac{dh_1}{dw_1}$

**(1) $dL / do_1$** $=-2(y_{true}-o_1)$

$Relation: L = (y_{true} - o_1)^2$

- $o_1 = y_{predict}$
- $dL / do_1 = -2(y_{true}-o_1)$


**(2) $do_1 / dh_1$**$=f'(z_{o1}) * w_5$

$Relation: o_1 = f(h1*w5 + h2*w6 + b3)$

- $z_{o1} = h1*w5 + h2*w6 + b3$
- $f(z_{o1}) = sigmoid(z_{o1}) = 1 / 1 + e^{-z_{o1}}$

$do_1 / dh_1 = \frac{do_1}{dz_{o1}} * \frac{dz_{o1}}{dh_1}$

$=f'(z_{o1}) * w_5$

**(3) $dh_1 / dw_1$** $=f'(z_{h1}) * x_1$
- $z_{h1} = x1*w1 + x2*w2 + b1$
- $f(z_{h1}) = sigmoid(z_{h1}) = 1 / 1 + e^{-z_{h1}}$

$=f'(z_{h1}) * x_1$

$=-2(y_{true}-o_1)$ * $(f'(z_{o1}) * w_5)$ * $(f'(z_{h1}) * x_1)$

## 4.2 Descent Gradient
$w_1 = w_1 - \alpha \frac{dL}{dw1}$

- (1) calculate derivatives of all weigits and bias
- (2) update every value, such as $w_1 = w_1 - \alpha \frac{dL}{dw1}$
- (3) loop (1)