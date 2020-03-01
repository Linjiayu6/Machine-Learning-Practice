# Fish age prediction

>[资料: 线性回归基础篇之预测鲍鱼年龄](https://cuijiahua.com/blog/2017/11/ml_11_regression_1.html)
>[资料: Jack-Cherish/Machine-Learning](https://github.com/Jack-Cherish/Machine-Learning/tree/master/Regression)


## 1. Recap

### 1.1 Dataset
- 1st column: $x_0 = 1$
- 2nd column: have 1 feautures $x_1$
- 3th column: $y$
<pre>
x0          x1          y
1.000000	0.067732	3.176513
1.000000	0.427810	3.816464
...
</pre>


### 1.2 Matrix Calculus
>[矩阵求导](https://blog.csdn.net/nomadlx53/article/details/50849941)
>[* Link: Coursera 4.7 正规方程及不可逆性](http://www.ai-start.com/ml2014/html/week2.html#header-n55)
>[* Link: 最小二乘法线性回归：矩阵视角](https://zhuanlan.zhihu.com/p/33899560)


## 2. Practice

### 2.1 Practice I
>[* Link: Practice I](https://github.com/Linjiayu6/Linear-Regression/wiki/%5BNote%5D-Practice-I)

`practice_1.py`

![practice_1](./imgs/practice_1.png)

**`存在欠拟合 underfit 情况`**


### 2.2 Practice II
TODO: Locally Weighted Linear Regression