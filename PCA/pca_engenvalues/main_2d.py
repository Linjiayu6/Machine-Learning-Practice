# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas

# 二维例子

def data_normalization ():
    x = np.array([2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1])
    y = np.array([2.4,0.7,2.9,2.2,3,2.7,1.6,1.1,1.6,0.9])
    # data normalization
    mu_x, mu_y = np.mean(x), np.mean(y)
    x = x - mu_x
    y = y - mu_y

    data = []
    for i in range(len(x)):
        data.append([x[i], y[i]])
    data = np.array(data)

    return data

def covariance (data):
    """
    data.T:
    [[ 0.69 -1.31  0.39  0.09  1.29  0.49  0.19 -0.81 -0.31 -0.71]
     [ 0.49 -1.21  0.99  0.29  1.09  0.79 -0.31 -0.81 -0.31 -1.01]]
    
    协方差结果
    [[0.61655556 0.61544444]
        [0.61544444 0.71655556]]
    样本x,y 之间的差距
    
    或者使用矩阵相乘，详见covariance.py
    """
    return np.cov(data.T) 

def eigendata (cov):
    # 特征值, 特征基, 特征向量
    eig_val, eig_vec = np.linalg.eig(cov)
    """
    lambda = array([0.0490834 , 1.28402771]),
    vector = array([[-0.73517866, -0.6778734 ], [ 0.6778734 , -0.73517866]])
    """
    return eig_val, eig_vec

def pca (eig_val, eig_vec, data):
    # 得到特征值和特征向量之后，我们可以根据特征值的大小，从大到小的选择K个特征值对应的特征向量
    # 特征值: array([0.0490834 , 1.28402771]
    # 特征向量: array([[-0.73517866, -0.6778734 ], [ 0.6778734 , -0.73517866]]))
    # 特征值和特征向量相关联
    eig_pairs = np.array([])
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
    eig_pairs.sort(reverse=True)
    # 根据特征值的最大值, 选择特征向量 1.28402771, [ 0.6778734 , -0.73517866]
    # 选择 [ 0.6778734 , -0.73517866]
    feature = eig_pairs[0][1]
    
    # PCA降维
    new_data_reduce = (np.dot(feature, data.T)).T

    return new_data_reduce
    
def draw(data, eig_val, eig_vec, new_data_reduce):
    # 原来的点
    plt.plot(data[:, 0], data[:, 1], 'o', color="red")
    # 特征向量
    print(eig_val, eig_vec)
    plt.plot([eig_vec[:,0][0],0],[eig_vec[:,0][1],0], color='green')
    plt.plot([eig_vec[:,1][0],0],[eig_vec[:,1][1],0], color='blue')
    
    # data经过特征向量, 矩阵乘积, 变为new_data
    # 相似矩阵，基变化
    new_data = (np.dot(eig_vec, data.T)).T
    plt.plot(new_data[:, 0], new_data[:, 1], '^')
    
    # new_data_reduce 降维数据
    # [-0.82797019  1.77758033 -0.99219749 -0.27421042 -1.67580142 -0.9129491 0.09910944  1.14457216  0.43804614  1.22382056]
    # y轴坐标 [2] 是为了区分和下面的点, 三角形的点都投影到了星星这些点
    # 红色的点, 投影到蓝色的线, 形成的点。
    # 这就是PCA,通过选择特征根向量，形成新的坐标系，然后数据投影到这个新的坐标系，在尽可能少的丢失信息的基础上实现降维。
    plt.plot(new_data_reduce, [2] * 10, '*')

    plt.show()
    
if __name__ == '__main__':
    # 数据归一化处理
    data = data_normalization()
    
    # 协方差矩阵
    cov = covariance(data)
    
    # 协方差的特征值和特征向量
    eig_val, eig_vec = eigendata(cov)
    
    # 基变化, 图中的三角点 (求解相似矩阵)
    # PCA
    new_data_reduce = pca (eig_val, eig_vec, data)
    
    # 绘图
    draw(data, eig_val, eig_vec, new_data_reduce)
    