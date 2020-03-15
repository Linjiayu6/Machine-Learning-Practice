# coding:utf-8

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html


def load_dataset (fileName):
    """
    load dataset
    return xArr, yArr
    """
    # open(fileName).readline(): '1.000000        0.067732        3.176513'
    # numFeat = 2
    # numFeat = len(open(fileName).readline().split('\t')) - 1
    fr = open(fileName)
    xArr, yArr = [], []

    for line in fr.readlines():
        lineArr = []
        # eg: ['1.000000', '0.067732', '3.176513']
        currentLine = line.strip().split('\t') 
        # X = [ [x0, x1], [x0, x1], .... ] (str -> float)
        for i in range(len(currentLine) - 1):
            lineArr.append(float(currentLine[i]))
        
        xArr.append(lineArr)
        # Y = [y1, y2, ... ]
        yArr.append(float(currentLine[-1]))

    return np.array(xArr), np.array(yArr)

def train (X, y_true):
    # init_weights = np.array([1, 1])
    # y = x0 + x1w1
    # y = np.dot(X, init_weights)
    # model
    model = LinearRegression().fit(X, y_true)

    # coefficients 系数
    print('(weights/theta)coef:', model.coef_)
    # ('(weights/theta)coef:', array([0.       , 1.7108288]))

    return model

def draw (model, X, y_true):
    plt.figure()
    plt.xlabel('X')
    plt.ylabel('y')
    plt.grid(True) # 显示网格
    
    # 原始数据 绘图
    plt.plot(X, y_true,'r.')
    
    # 预测函数绘图
    plt.plot(X, model.predict(X), color='blue', linewidth=3)
    
    plt.show()

if __name__ == '__main__':
    X, y_true = load_dataset('dataset.txt')
    model = train(X[:-50], y_true[:-50])
    # test set
    X_test, y_true_test = X[-50:], y_true[-50:]
    y_test_predict = model.predict(X_test)

    print('J(test):', np.mean((y_test_predict - y_true_test) ** 2))
    print('J(test)  model.score:', model.score(X_test, y_test_predict))

    draw(model, X, y_true)
    # print(X, y)
