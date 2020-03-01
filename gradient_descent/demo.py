# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

def draw():
    # 范围是(-1, 1);个数是50
    x = np.linspace(-1, 5, 50)
    y = - x ** 2 + 4 * x
    
    plt.figure()
    # set style
    plt.plot(x, y, c = 'pink')
    plt.show()


# y = -x^2 + 4x
def math_derivative(x):
    return -2 * x + 4

def gradient_descent():
    """
    x_new = x_old + alpha * derivative_by_x_old
    alpha: learning rate

    abs(x_new - x_old): distance limit to 0 (这里设定为0.000000000001)
    """
    # 初始化值设定, 从0开始
    x_new = 0
    x_old = -1
    
    # learning rate
    alpha = 0.01
    
    # 0.00000000000001 本应该是为无穷小的情况
    while abs(x_new - x_old) > 0.00000000000001:
        x_old = x_new
        x_new = x_old + alpha * math_derivative(x_old)
    return x_new, List

x_new, List = gradient_descent()
# 1.9999999999995068, 很接近2
print('极限值, 应该是2, 梯度模拟接近: ', x_new)

draw()