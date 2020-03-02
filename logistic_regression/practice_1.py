# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def loadData ():
    fr = open('./dataset.txt')
    """
    x           y           分类
    -0.017612	14.053064	0
    """
    data_mat, label_arr = [], []
    for linestr in fr.readlines():
        lineArr = linestr.strip().split()
        # data_arr = [ [1, x1, x2 ], [...], ... ]
        data_mat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        # label_arr [xx, xx, ...] 
        label_arr.append(int(lineArr[2]))   
    fr.close()
    return data_mat, label_arr

def draw (data_mat, label_arr, weights_3_1):
    dataArr = np.array(data_mat) # 转换成numpy的array数组
    m = dataArr.shape[0] # 样本的个数                                    
    positive_x, positive_y = [], []
    negative_x, negative_y = [], []
    for i in range(m):
        x = dataArr[i][1]
        y = dataArr[i][2] # 对数据来说, 离散点是个坐标
        if label_arr[i] == 1: # 正样本
            positive_x.append(x)
            positive_y.append(y)
        else:  #0为负样本
            negative_x.append(x)
            negative_y.append(y)

    fig = plt.figure()
    ax = fig.add_subplot(111) 

    # 画直线
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights_3_1[0][0] - weights_3_1[1][0] * x) / weights_3_1[2][0]
    ax.plot(x, y)

    # 散列图                                   
    ax.scatter(positive_x, positive_y, s = 20, c = 'red', marker = 's',alpha=.5)
    ax.scatter(negative_x, negative_y, s = 20, c = 'green',alpha=.5)            
    plt.title('positive: red; negative: green')                                                
    plt.show()
    
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train(data_mat, label_arr):
    alpha = 0.001 # learning rate
    loopnum = 1000 # repeat counts
    
    X_100_3 = np.mat(data_mat) # (100 * 3) [ [1, xx, xxx], [1. xx, xxx ], ... ]
    y_100_1 = np.mat(label_arr).transpose() # (100 * 1) [[y1], [y2], ... ]
    
    """
    (1) z(x) = w0x0 + w1x1 + w2x2 + ... = X * w
    (2) h(x) = g(z) = sigmoid(z)
    (3) dW_xj = (y - h(x)) * xj
    (4) weights = weights + alpha * dW_xj
    (5) repeat (1) => loopnum
    """
    m, n = np.array(X_100_3).shape # m=100, n=3
    weights_3_1 = np.ones((n, 1)) # [ [1], [1], [1] ]
    
    for i in range(loopnum):
        # (1) z
        z_100_1 = X_100_3 * weights_3_1
        # (2) h
        h_100_1 = sigmoid(z_100_1)
        # (3) derivative
        dW_3_1 = X_100_3.T * (y_100_1 - h_100_1)
        # (4) weights
        weights_3_1 += alpha * dW_3_1
    print('weights: ', weights_3_1)
    return weights_3_1
    
if __name__ == "__main__":
    # 加载数据
    data_mat, label_arr = loadData()
    weights_3_1 = train(data_mat, label_arr)

    # 绘图
    draw(data_mat, label_arr, weights_3_1)
