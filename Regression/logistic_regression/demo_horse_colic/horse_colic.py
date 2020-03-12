# -*- coding:utf-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt
def draw (J_arr):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(len(J_arr)), J_arr, '.-')
    plt.title('Cost Function & loop')                                                
    plt.show()
    
def loadData (txt):
    # 训练集合载入
    fr = open(txt)
    X, y_arr = [], []
    for linestr in fr.readlines():
        lineArr = linestr.strip().split()
        tempArr = [1.0]
        for i in range(len(lineArr) - 1):
            tempArr.append(float(lineArr[i]))
        y_arr.append(float(lineArr[-1]))
        X.append(tempArr)
        
    x_mean = np.mat(X).mean(0)

    # X(299, 22) , y_arr(299,)
    return np.mat(X) - x_mean, np.array(y_arr)

def sigmoid (z):
    # sigmoid function
    return 1 / (1 + np.exp(-z))

def classifer (X, y_arr, weights):
    # 分类器: z = X * w
    z = X * weights
    y_predict = np.array(sigmoid(z)).T[0]
    m = len(y_arr)
    error = []
    for i in range(m):
        if y_predict[i] >= 0.5:
            if y_arr[i] != 1:
                error.append(i)
        else:
            if y_arr[i] != 0:
                error.append(i)
    return float(len(error)) / float(m)

def train (X, y_arr):
    # 0. init
    # alpha = 0.01
    loop = 500

    # 1. weights
    m, n = X.shape # m = 299 data, n = 22 features [w0, w1, ...]
    weights = np.ones((n, 1))
    
    # 绘图J(\theta) 和 迭代次数
    J_arr = []
    
    for i in range(loop):
        # alpha = 0.01 + 5 / (i + 1)
        alpha = 0.1

        # 2. z = X * weights = sum( w1x1 + w2x2 + ... ) => z = (299 * 22) * (22 * 1) = (299 * 1)
        z = X * weights
        
        # 3. h(x) = g(z) = sigmoid(z)
        hypothesis = sigmoid(z)
        
        # 绘图J(\theta) 和 迭代次数
        J_arr.append(costfunction(hypothesis, np.mat(y_arr).T))
        
        # 4. derivatives
        # dJ/dw2 = (h - y) x2  Matrix = X.T * (h - y) (299 * 22).T * (299 * 1) = (22 * 1)
        derivative = X.T * (hypothesis - np.mat(y_arr).T)
        
        # 5. GD
        weights = weights - (alpha * derivative) / m
    return weights, J_arr

# 解决RuntimeWarning: divide by zero encountered in log2问题
def costfunction (y_arr, y_predict):
    epsilon = 1e-5
    y = np.array(np.mat(y_arr).T)
    h = np.array(y_predict)
    return np.average(- y * np.log2(h + epsilon) - (1 - y) * np.log2(1 - h + epsilon))
    
def evaluate (X, y_arr, weights, printText):
    y_predict = sigmoid(X * weights)
    J_sum = costfunction(y_arr, y_predict)
    J = np.average(J_sum / y_arr.shape[0])
    print(printText, 'J(theta): ', J)

if __name__ == "__main__":
    # (1) load training set
    X, y_arr = loadData('./data/trainingset.txt')

    # (2) train
    weights, J_arr = train(X, y_arr)
    
    # 损失函数和迭代次数关系
    draw(J_arr)
    
    # (3) evaluate costfn by training set
    evaluate(X, y_arr, weights, 'training set')
    
    # (4) evaluate costfn by test set
    X_test, y_test_arr = loadData('./data/testset.txt')
    evaluate(X_test, y_test_arr, weights, 'test set')
    
    # (5) error rate
    err_rate = classifer(X, y_arr, weights)
    print('Training dataset: error rate', err_rate * 100, '%')

    err_rate = classifer(X_test, y_test_arr, weights)
    print('Test dataset: error rate', err_rate * 100, '%')
    