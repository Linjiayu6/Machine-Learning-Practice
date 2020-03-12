# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import math

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
    # w0x0 + w1x1 + w2x2 = w0x0 + w1x1 + w2y (x2其实是y的坐标值)
    # -(w0x0 + w1x1) = w2y => y = -(w0x0 + w1x1) / w2
    y = - (weights_3_1[0][0] + weights_3_1[1][0] * x) / weights_3_1[2][0]
    ax.plot(x, y)

    # 散列图                                   
    ax.scatter(positive_x, positive_y, s = 20, c = 'red', marker = 's',alpha=.5)
    ax.scatter(negative_x, negative_y, s = 20, c = 'green',alpha=.5)            
    plt.title('positive: red; negative: green')                                                
    plt.show()
   
def draw_weights_loopnum (weights_loopnum_mat):
    """
    [ [0.98786192 0.99428432 0.82942775]
      [0.97577096 0.98853428 0.65942642]
      ... ]
    """
    # 画布分成3个区域 3 * 1 axs[0][0]表示第一行第一列
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=False, sharey=False, figsize=(20, 10))
    # 0 - 循环个数, 中间隔一位
    x1 = np.arange(0, len(weights_loopnum_mat), 1)
    # w0
    axs[0].plot(x1, weights_loopnum_mat[:,0])
    # axs0_ylabel_text = axs[0][0].set_ylabel(u'w0')
    # plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    # w1
    axs[1].plot(x1, weights_loopnum_mat[:,1])
    # axs1_ylabel_text = axs[1][0].set_ylabel(u'w1')
    # plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    # w2
    axs[2].plot(x1, weights_loopnum_mat[:,2])
    # axs2_ylabel_text = axs[2][0].set_ylabel(u'w2')
    # plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')
    
    plt.show()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train(data_mat, label_arr):
    alpha = 0.01 # learning rate
    loopnum = 500 # repeat counts
    
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
    
    # 绘图使用: weights数据 和 迭代次数 的关系
    weights_loopnum_arr = np.array([])

    # gradient descent
    for i in range(loopnum):
        # (1) z
        z_100_1 = X_100_3 * weights_3_1
        # (2) h
        h_100_1 = sigmoid(z_100_1)
        # (3) derivative
        dW_3_1 = X_100_3.T * (h_100_1 - y_100_1)
        # (4) weights
        weights_3_1 -= alpha * dW_3_1 / m
        
        # 绘图使用: weights数据 和 迭代次数 的关系
        weights_loopnum_arr = np.append(weights_loopnum_arr, weights_3_1)

    """
      例如: weights_loopnum_arr = [1, 2, 3, 4, 5, 6], 循环2次
      loopnum = 2, n = 3
      分为2行3列的矩阵
    """
    weights_loopnum_mat = weights_loopnum_arr.reshape(loopnum, n)
    return weights_3_1, weights_loopnum_mat

def evaluate(data_mat, label_arr, weights_3_1):
    # 评估模型
    z_100_1 = np.mat(data_mat) * weights_3_1
    h_100_1 = sigmoid(z_100_1)
    
    # C = -ylogh(x) - (1 - y)log(1 - h(x))
    z = np.array(z_100_1)
    h = np.array(h_100_1).flatten()
    
    def costfunction (y_predict, y_true):
        return -y_true * math.log(y_predict, 2) - (1 - y_true) * math.log(1 - y_predict, 2)

    loss_arr, error_arr = [], []
    for i in range(len(label_arr)):
        y_true = label_arr[i]
        y_predict = h[i]
        loss_value = costfunction(y_predict, y_true)
        loss_arr.append(loss_value)
    print('Successful Rate:', (1 - sum(loss_arr) / len(label_arr)) * 100, '%')
    
if __name__ == "__main__":

    # 加载数据
    data_mat, label_arr = loadData()
    weights_3_1, weights_loopnum_mat = train(data_mat, label_arr)

    # 绘图
    # draw(data_mat, label_arr, weights_3_1)
    
    # 绘图: weights数据 和 迭代次数 的关系
    draw_weights_loopnum(weights_loopnum_mat)
    
    # 评估模型
    evaluate(data_mat, label_arr, weights_3_1)
