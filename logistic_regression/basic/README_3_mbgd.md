
# mini-Batch Gradient Descent 小批量梯度下降

## 1. BGD / GD 解决思路

```
(1) z = w0x0 + w1x1+ w2x2 = X.T * w
(2) Model: h(x) = g(z) = sigmoid function = 1 / 1 + e^-z
(3) J(w) = 1 / m * sum( -ylog(h(x)) - (1-y)log(1-h(x) )
(4) dJ(w)/dw = 1/m * (h(x) - y) x = 1/m * X.T * (H(x) - Y)
    w = w - alpha * dJ(w)/dw
    Repeat: (1)
```

## 2. SGD 解决思路
```
(1) z = w0x0 + w1x1+ w2x2 = X.T * w
(2) Model: h(x) = g(z) = sigmoid function = 1 / 1 + e^-z
(3) J(w) = 1 / m * sum( -ylog(h(x)) - (1-y)log(1-h(x) )
(4) dJ(w)/dw = (h(x) - y) x (随机挑选某一项, 非全部)
    w = w - alpha * dJ(w)/dw
    Repeat: (1)
```

## 3. MSGD 解决思路
```
(1) z = w0x0 + w1x1+ w2x2 = X.T * w
(2) Model: h(x) = g(z) = sigmoid function = 1 / 1 + e^-z
(3) J(w) = 1 / m * sum( -ylog(h(x)) - (1-y)log(1-h(x) )
(4) dJ(w)/dw = 1/n * X.T * (H(x) - Y) (选取一组长, 数量为n, 非全部)
    w = w - alpha * dJ(w)/dw
    Repeat: (1)
```

## 4. 问题
- learning rate 如何影响?
- mini-batch 怎么分? 分多少份儿?
- 图像的波动代表什么?
- 收敛速度是否有提升?