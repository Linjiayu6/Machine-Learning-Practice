# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
# linear algebra
import numpy.linalg as la
# 分割数据的模块, 把数据分成training set, cv, test set...
from sklearn import datasets
# 图形处理
from skimage import io

def getImageFromFile(filename):
    img = io.imread(filename, as_grey=True)
    # img.shape (546, 600)
    return np.mat(img)

def draw(imgMat):
    plt.imshow(imgMat, cmap=plt.cm.gray)
    plt.show()
    
def compression(imgMat, d):
    # SVD V初始化基向量, U变换后基向量, S从V到U变换中的模(倍数)
    U, Sigma, VT = la.svd(imgMat)
    
    # 选择压缩的维度d
    # 变换后的基向量
    U_d = U[:, 0:d]
    # np.dialog, 创建一个对角矩阵
    Sigma_d = np.diag(Sigma[:d])
    # 变换前的基向量
    VT_d = VT[0:d, :]

    return U_d * Sigma_d * VT_d


if __name__ == '__main__':
    # 读取图像
    imgMat = getImageFromFile('/Users/linjiayu/meituan/SELF_LEARN/Machine-Learning-Practice/PCA/imgs/image.jpg')
    # 绘制原图
    # draw(imgMat)

    # 压缩图像
    newImgMat = compression(imgMat, 20)
    # 绘制压缩后的图像
    draw(newImgMat)