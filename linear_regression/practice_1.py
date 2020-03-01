# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

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

    return xArr, yArr


def matplot (xArr, yArr, X_copy, Y_predict):
    # Xaxis = [0.067732, ...] Yaxis = [3.176513]
    Xaxis, Yaxis = [], []
    # xitem = [ [1.0, 0.067732], ... ] yitem = [3.176513, ... ]
    for i in range(len(yArr)):
       Xaxis.append(xArr[i][1])
       Yaxis.append(yArr[i])

    fig = plt.figure()
    # There is only one subplot or graph
    ax = fig.add_subplot(111) 
    # s: fontsize c: color alpha(transparent)
    ax.scatter(Xaxis, Yaxis, s = 20, c = 'pink', alpha = 0.5)
    
    # X_copy: matrix X_copy.flatten() => array
    ax.plot(X_copy.flatten().A[0], Y_predict.flatten().A[0], c = 'red') 

    plt.title('dataset')
    plt.xlabel('X(features)')
    plt.ylabel('Y')
    plt.show()
    
def cost_function (y_predict, y_true):
    # y_predict, y_true => m * 1 matrix
    # 平方和 = A.T * A
    A = (y_predict - y_true)
    SUM = A.T * A

    # m个数
    m = y_predict.shape[0]
    
    return SUM / (2 * m)

def train (xArr, yArr):
    # Y => Y.T
    X, Y = np.mat(xArr), np.mat(yArr).T

    # Matrix Caculus T=transpose I=inverse
    dW = (X.T * X).I * (X.T * Y)
    return dW
    
if __name__ == '__main__':
    # 载入数据
    xArr, yArr = load_dataset('dataset.txt')
    
    # 训练: 得到矩阵导数
    dW = train(xArr, yArr) # dW = [[3.00774324] [1.69532264]]
    
    X_copy = np.mat(xArr).copy()
    # [ [1, 0.3], [1, 0.4] ] 排序是为了方便matplot图像输出
    X_copy.sort(0)
    
    # 预测: y = X * w
    Y_predict = X_copy * dW
    
    # 拟合效果: 比较真实值和预测值得相关性 np.corrcoef(Y_predict, Y_train)
    convex = np.corrcoef(Y_predict.T, np.mat(yArr))
    print('[Result of fitting]: ', convex)
    """
    0.13653777 有弱正相关
    [
        [1.         0.13653777]
        [0.13653777 1.        ]
    ]
    """
    
    # 损失值
    loss = cost_function(Y_predict, np.mat(yArr).T)
    print('[Loss Value]: ', loss) # 0.21483169
    
    # 图像
    # matplot(xArr, yArr, X_copy[:, 1], Y_predict)
