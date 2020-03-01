# 1. demo.py
> [梯度上升算法例子](https://cuijiahua.com/blog/2017/11/ml_6_logistic_1.html)


问题: 求解极大值的例子
例子: $ y = -x^2 + 4x$
导数: $ dy/dx = -2x + 4 = 0$ 当$x = 2$为极大值

```
x_new = x_old + a * dy/dx_old

a: learningrate
```
