# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def loadData ():
    fr = open('./dataset')
    """
    x           y           分类
    -0.017612	14.053064	0
    """
    data_arr, label_arr = [], []
    for linestr in fr.readlines:
        lineArr = linestr.strip().split()
        data_arr.append([1.0, float(lineArr[0]), float(lineArr[1])])
        label_arr.append(float(lineArr[2]))   
    fr.close()
    return data_arr, label_arr

def draw (data_arr, label_arr):
    # 转换成numpy的array数组

data_arr, label_arr = loadData()