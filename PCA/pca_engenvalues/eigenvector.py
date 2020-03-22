# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# Ax = lambdax
# lambdaI 是对角矩阵
# det(A - lambdaI) = 0 他们是在一个平面上或者是一条方向线上的...
# 因为这样才可以去 降维


x1 = [7, 2]
x2 = [1, 4]
data = np.array([x1, x2])
# 求协方差矩阵的特征根和特征向量
cov = np.array([[18, -6], [-6, 2]])
eig_val, eig_vec = np.linalg.eig(cov)
"""
(array([20.,  0.]), array([[ 0.9486833 ,  0.31622777],
       [-0.31622777,  0.9486833 ]]))

特征值为: 20. 或 0.
特征向量为: array([[ 0.9486833 ,  0.31622777], [-0.31622777,  0.9486833 ]]))

特征值lambda = 20 => 特征向量为: [ 0.9486833 ,  0.31622777] 
规范化后: 0.9486833^2 + 0.31622777^2 = 1

特征值lambda = 0 => 特征向量为: [-0.31622777,  0.9486833 ]
规范化后: -0.31622777^2 + 0.9486833^2 = 1

如何选择lambda?
选择lambda最大。 20 / 20 + 0 = 100%。
说明降维后, 可以很好的保留原来的数据，保真度是100%。
"""
# 特征向量 * data.T
new_data = (eig_vec * data.T)

# 绘图
plt.plot(new_data[:, 0], new_data[:, 1], '^', color='blue')
plt.plot(data[:, 0], data[:, 1], 'o')

# 手动计算出来的特征向量(并未规范化的)
plt.plot([1, 0], [3, 0], color='green')
plt.plot([3, 0], [-1, 0], color='green')

# np计算出来的特征向量
plt.plot([eig_vec[:,0][0],0],[eig_vec[:,0][1],0],color='red')
plt.plot([eig_vec[:,1][0],0],[eig_vec[:,1][1],0],color='red')
plt.show()