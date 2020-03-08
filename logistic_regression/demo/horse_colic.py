# -*- coding:utf-8 -*-

import math
import numpy as np
# import matplotlib.pyplot as plt

def loadData ():
    # 训练集合载入
    fr = open('./data/trainingset.txt')
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

def classifer (z):
    # 分类器: z = X * w
    probability = sigmoid(z)
    if probability >= 0.5:
        return 1
    else:
        return 0

def train (X, y_arr):
    # 0. init
    alpha = 0.01
    loop = 100

    # 1. weights
    m, n = X.shape # m = 299 data, n = 22 features [w0, w1, ...]
    weights = np.ones((n, 1))
    
    for i in range(loop):
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
        fn_2 = np.log2(1 - y_predict)
        return np.multiply(np.array(y_1), np.array(fn_1)) + np.multiply(np.array(y_2), np.array(fn_2))
    
    print(costfunction(y_arr, y_predict)[:5])
    J = costfunction(y_arr, y_predict) / y_arr.shape[0]
    # print('Successful Rate:', (1 - sum(np.array(J).T) / len(y_arr)) * 100, '%')

X, y_arr = loadData()
weights = train(X, y_arr)

evaluate(X, y_arr, weights)