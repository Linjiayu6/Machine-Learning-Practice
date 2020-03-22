# -*- coding:utf-8 -*-

import numpy as np
from sklearn.decomposition import PCA

# 多维度的例子
X = np.array([[-1, 1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

# 自己写的例子
def pca (X, k):
    # 1. normalization
    m, n = X.shape # m: samples, n: features
    mean = np.array([np.mean(X[:, i]) for i in range(n)]) 
    # mean eg: [0., 0.33333333]
    X_normalization = X - mean
    
    # 2. covariance X.T * X 矩阵相乘
    Covariance = np.dot(X_normalization.T, X_normalization)
    
    # 3. eigenvectors eigenvalues 计算特征值和特征向量
    eig_val, eig_vec = np.linalg.eig(Covariance)
    
    # 4. select abs(eig_val)最大的eigenvectors
    eig_val_vec = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(n)]
    eig_val_vec.sort(reverse=True)
    
    # 5. 选择k维度
    features = np.array([element[1] for element in eig_val_vec[:k]])
    # [[0.8549662  0.51868371]]
    print(features)
    
    # 6. 压缩后的新数据
    """
    [[-0.50917706]
    [-2.40151069]
    [-3.7751606 ]
    [ 1.20075534]
    [ 2.05572155]
    [ 3.42937146]]
    """
    new_X = np.dot(X_normalization, features.T)
    print(new_X)
    
pca(X, 1)

# 库的例子
sklearn_pca = PCA(n_components = 1)
sklearn_pca.fit(X)
print(sklearn_pca.transform(X))