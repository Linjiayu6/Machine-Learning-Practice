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
def draw (data_mat, classify_arr, w_3_1):
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

    ax1 = plt.subplot(111)
    # 直线 -(w0x0 + w1x1) / w2 = y
    x = np.arange(-3.0, 3.0, 0.1)
    w0, w1, w2 = w_3_1
    y = (-w0 - w1 * x) / w2
    ax1.plot(x, y)
    # 散列图
    ax1.scatter(positive_x_arr, positive_y_arr, s = 20, c = 'red', marker = 's',alpha=.5)
    ax1.scatter(negative_x_arr, negative_y_arr, s = 20, c = 'green',alpha=.5)

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

# 评估
def evaluate(data_mat, classify_arr, w_3_1):
    z_100_1 = data_mat * np.mat(w_3_1)
    h_100_1 = sigmoid(z_100_1)
    
    # C = -ylogh(x) - (1 - y)log(1 - h(x))
    z = np.array(z_100_1)
    h = np.array(h_100_1).flatten()
    
    def costfunction (y_predict, y_true):
        return -y_true * math.log(y_predict, 2) - (1 - y_true) * math.log(1 - y_predict, 2)

    loss_arr, error_arr = [], []
    for i in range(len(classify_arr)):
        y_true = classify_arr[i]
        y_predict = h[i]
        loss_value = costfunction(y_predict, y_true)
        loss_arr.append(loss_value)
    print('Successful Rate:', (1 - sum(loss_arr) / len(classify_arr)) * 100, '%')

# 训练
def train (data_mat, classify_arr):
    classify_matrix = np.mat(classify_arr).transpose()

    m, n = np.array(data_mat).shape # 100 * 3 matrix, 3 features (w0,w1,w2), 100 data
    # alpha = 0.01 # 学习速率
    loop_num = 150 # 训练次数
    
    group = 50 # 分成20组
    num_one_group = m / group # 每组的个数 eg: loop_num = 20, num_one_group = 5

    w_3_1 = np.ones((n, 1)) #[ [1], [1], [1]]
    
    weights_loopnum_arr = []

    for i in range(m):
        for j in range(group):
            # (1) z: z_p_1 
            # random_num = int(random.uniform(0, len(data_mat) - num_one_group))
            start = j * num_one_group
            end = start + num_one_group
            min_data_p_3 = np.mat(data_mat[start : end])
            z_p_1 = min_data_p_3 * w_3_1
            # (2) h_p_1
            h_p_1 = sigmoid(z_p_1)

            # (3) dJ(W) / dw 
            y_p_1 = classify_matrix[start : end]
            error_p_1 = h_p_1 - y_p_1

            dJW_dw = (min_data_p_3.T * error_p_1) / num_one_group
            # (4) w_3_1
            alpha = 0.01 + 10 / (1 + i + j)
            # alpha = 0.01
            w_3_1 -= alpha * dJW_dw
            
            # 绘图使用: weights数据 和 迭代次数 的关系
            weights_loopnum_arr = np.append(weights_loopnum_arr, w_3_1)
    weights_loopnum_mat = weights_loopnum_arr.reshape(m * group, n)
    return w_3_1, weights_loopnum_mat

if __name__ == "__main__":
    # 1. 加载数据
    data_mat, classify_arr = load_data()

    # 2. 训练
    w_3_1, weights_loopnum_mat = train(data_mat, classify_arr)
    
    # 3. 预测
    evaluate(data_mat, classify_arr, w_3_1)
    
    # 4. 绘图
    draw(data_mat, classify_arr, w_3_1)
    draw_weights_loopnum(weights_loopnum_mat)
    