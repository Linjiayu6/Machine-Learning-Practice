# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import random
import math

# 加载数据
def load_data ():
    fr = open('./dataset.txt')
    data_mat, classify_arr = [], []
    
    for line_str in fr.readlines():
        x, y, classifier = line_str.strip().split()
        data_mat.append([1, float(x), float(y)])
        classify_arr.append(int(classifier))

    fr.close()    
    return data_mat, classify_arr

# 绘图 (数据散列点 和 预测的逻辑回归图)
def draw (data_mat, classify_arr, w_1_3):
    m, n = np.array(data_mat).shape
    positive_x_arr,  positive_y_arr = [], []
    negative_x_arr,  negative_y_arr = [], []

    for i in range(m):
        data = data_mat[i]
        # positive
        if (classify_arr[i] == 1):
            positive_x_arr.append(data[1])
            positive_y_arr.append(data[2])
        else:
            # negative
            negative_x_arr.append(data[1])
            negative_y_arr.append(data[2])
    
    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # 直线 -(w0x0 + w1x1) / w2 = y
    x = np.arange(-3.0, 3.0, 0.1)
    w0, w1, w2 = w_1_3
    y = (-w0 - w1 * x) / w2
    ax.plot(x, y)

    # 散列图
    ax.scatter(positive_x_arr, positive_y_arr, s = 20, c = 'red', marker = 's',alpha=.5)
    ax.scatter(negative_x_arr, negative_y_arr, s = 20, c = 'green',alpha=.5)
    
    plt.title('positive: red; negative: green')
    plt.show()

# 绘图 (迭代次数和收敛效率的关系图)
def draw_weights_loopnum (weights_loopnum_mat):
    # 画布分成3个区域 3 * 1 axs[0][0]表示第一行第一列
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=False, sharey=False, figsize=(20, 10))
    x1 = np.arange(0, len(weights_loopnum_mat), 1)
    # w0
    axs[0].plot(x1, weights_loopnum_mat[:,0])
    axs[0].set_ylabel(u'w0')
    # w1
    axs[1].plot(x1, weights_loopnum_mat[:,1])
    axs[1].set_ylabel(u'w1')
    # w2
    axs[2].plot(x1, weights_loopnum_mat[:,2])
    axs[2].set_ylabel(u'w2')

    plt.show()

# sigmoid
def sigmoid (z):
    return 1 / (1 + np.exp(-z))

# 训练
def train (data_mat, classify_arr):
    m, n = np.array(data_mat).shape # m = 100(样本数量) n = 3(有三个权重)
    # 初始化 weights [1. 1. 1.]
    w_1_3 = np.ones(n)
    # 学习次数
    loopnum = 500
    # 学习率 alpha = 0.02

    data = np.array(data_mat)
    # 绘图使用: 权重和迭代次数的关系
    weights_loopnum_arr = []

    # 随机梯度下降算法 ((1) 并非对矩阵操作, 是对数组(一组数据) 操作 (2) 学习率也有所变化)
    for i in range(loopnum):
        # 暂存列表(目的: 每次训练样本不重复)
        _templist = list(range(m)) # 例如 m = 5 _templist: [0, 1, 2, 3, 4]

        # m = 100个样本, 对每个样本进行迭代计算权重值, 并每次训练样本不重复
        for j in range(m):
            # 降低alpha的大小，每次减小 1 / (j + i), i和j有可能为0会出错的所以(i + j + 1)
            # alpha = 0.01 + 10 / (1 + j + i)
            alpha = 0.015
            
            # 从_templist中, 选择个随机数
            random_index = int(random.uniform(0, len(_templist))) # 从0-_templist, 随机选个index
            random_index_in_templist = _templist[random_index] # 在暂存列表中的某个值
            x_1_3 = data[random_index_in_templist] # 选取当前值
            y_1_1 = classify_arr[random_index_in_templist] # 当前y_true值

            # 1. z 一个值 这里容易出错 [1,2,3]*[1,2,3] = [1,4,9] = sum(14)
            z_1_1 = sum(x_1_3 * w_1_3)
            # 2. h(x) = g(z)
            h_1_1 = sigmoid(z_1_1)
            # 3. derivative
            dw_1_3 = x_1_3 * (h_1_1 - y_1_1)
            # 4. weight
            w_1_3 = w_1_3 - alpha * dw_1_3
            
            # 将w_3_1 放入至weights_loopnum_arr数组中
            weights_loopnum_arr = np.append(weights_loopnum_arr, w_1_3, axis=0)
            del(_templist[random_index])
            
    # 处理weights_loopnum_arr
    weights_loopnum_mat = weights_loopnum_arr.reshape(loopnum * m, n)
    return w_1_3, weights_loopnum_mat

# 评估
def evaluate(data_mat, label_arr, w_1_3):
    z_100_1 = data_mat * np.mat(w_1_3).T
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
    # 数据, 真实分类值
    data_mat, classify_arr = load_data()

    # 权重, 权重和循环次数的矩阵
    w_1_3, weights_loopnum_mat = train(data_mat, classify_arr)

    # 绘图
    draw(data_mat, classify_arr, w_1_3)

    # 绘关系图
    draw_weights_loopnum(weights_loopnum_mat)
    
    # 评估
    evaluate(data_mat, classify_arr, w_1_3)