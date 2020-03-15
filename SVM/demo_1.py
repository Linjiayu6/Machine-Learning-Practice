# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from sklearn import svm, datasets
import pandas as pd

# https://zhuanlan.zhihu.com/p/71074401

def load_data():
    # load data
    # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
    iris = datasets.load_iris()
    """
    features: 4
    # print(iris.feature_names)
    ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

    # print(iris.target_names)
    ['setosa' 'versicolor' 'virginica']
    X = iris.data
    y = iris.target
    """

    # features: select 2 features
    X = iris.data[:, :2]
    # output: 3 ['setosa' 'versicolor' 'virginica']
    y = iris.target
    return X, y

def train():
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    # Support Vector Machine for Regression implemented using libsvm.
    
    # kernel: linear
    # kernel: rbf
    svc = svm.SVC(kernel='rbf', C=1, gamma='auto').fit(X, y)
    return svc

def draw(X, svc):
    plt.subplot(1, 1, 1)
    plt.xlabel('sepal length (cm)')
    plt.ylabel('sepal width (cm)')
    
    # 散列点
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)

    # 创建一个网格来进行可视化
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = (x_max / x_min) / 100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h))
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    # 绘图
    plt.show()

if __name__ == '__main__':
    X, y = load_data()
    svc = train()
    draw(X, svc)