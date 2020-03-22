# -*- coding:utf-8 -*-

# 协方差用于衡量两个变量的总体误差
# 目标: 计算变量间的相关性, eg: 2维, 目标: cov(x,y) = 0

import numpy as np

x1 = [7, 2]
x2 = [1, 4]
data = np.array([x1, x2])
print(data)

# 计算1
cov1 = np.cov(data.T)
print('covariance 1 np.cov', cov1)
"""
[[18. -6.]
 [-6.  2.]]
"""

# 或者: 计算2 矩阵相乘
data = data.T
data[0] = data[0] - np.mean(data[0])
data[1] = data[1] - np.mean(data[1])

# 矩阵相乘
cov2 = np.dot(data, data.T)
print('covariance 2', cov2)