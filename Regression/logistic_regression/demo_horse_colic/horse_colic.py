# -*- coding:utf-8 -*-

import math
import numpy as np
# import matplotlib.pyplot as plt

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

    # X(299, 22) , y_arr(299,)
    return np.mat(X), np.array(y_arr)

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
    loop = 100000

    # 1. weights
    m, n = X.shape # m = 299 data, n = 22 features [w0, w1, ...]
    weights = np.ones((n, 1))
    
    for i in range(loop):
        # alpha = 0.01 + 3 / (i + 1)
        alpha = 0.001 + 3 / (i + 1)

        # 2. z = X * weights = sum( w1x1 + w2x2 + ... ) => z = (299 * 22) * (22 * 1) = (299 * 1)
        z = X * weights
        
        # 3. h(x) = g(z) = sigmoid(z)
        hypothesis = sigmoid(z)
        
        # 4. derivatives
        # dJ/dw2 = (h - y) x2  Matrix = X.T * (h - y) (299 * 22).T * (299 * 1) = (22 * 1)
        derivative = X.T * (hypothesis - np.mat(y_arr).T)
        
        # 5. GD
        weights -= (alpha / m) * derivative

    return weights

def evaluate (X, y_arr, weights):
    y_predict = sigmoid(X * weights)
    
    def costfunction (y_arr, y_predict):
        y_1 = - np.mat(y_arr).T
        fn_1 = np.log2(y_predict)
        y_2 = - np.mat((1 - y_arr)).T
        # 遇到log(0) 负无穷大情况, 补0 ???
        fn_2 = np.log2(1 - y_predict)
        return np.multiply(np.array(y_1), np.array(fn_1)) + np.multiply(np.array(y_2), np.array(fn_2))
    
    J = costfunction(y_arr, y_predict)
    # inf情况 infinite, 补0
    inf_index = np.isinf(J)
    J[inf_index] = 0
    J = J / y_arr.shape[0]

    rate = (1 - J.sum() / len(y_arr)) * 100
    print('Successful Rate:', rate, '%')

if __name__ == "__main__":
    # (1) load training set
    X, y_arr = loadData('./data/trainingset.txt')
    
    # (2) train
    weights = train(X, y_arr)
    
    # (3) evaluate costfn by training set
    evaluate(X, y_arr, weights)
    
    # (4) evaluate costfn by test set
    X_test, y_test_arr = loadData('./data/testset.txt')
    evaluate(X_test, y_test_arr, weights)
    
    # (5) error rate
    err_rate = classifer(X, y_arr, weights)
    print('Training dataset: error rate', err_rate)

    err_rate = classifer(X_test, y_test_arr, weights)
    print('Test dataset: error rate', err_rate)
    