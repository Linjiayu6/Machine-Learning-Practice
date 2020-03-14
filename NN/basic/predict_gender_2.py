# -*- coding:utf-8 -*-

import numpy as np

def sigmoid(z):
    # 1 / 1 + e^(-z)
    return 1 / (1 + np.exp(-z))

def derivative_sigmoid(a):
    # derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    # fx = sigmoid(z)
    # return fx * (1 - fx)
    return a * (1 - a)

class NN():
    def __init__ (self):
        self.weights = np.random.rand(6)
        self.bias = np.random.rand(3)
        self.alpha = 0.1
        self.loopnum = 1000
        
    def predict (self, X, y):
        m = np.mat(X).shape[0]
        y_predict = []
        for i in range(m):
            a = self.forward_propagation(X[i])
            y_predict.append(a[-1][0])
        
        J = ((y_predict - y) ** 2).mean()
        print('cost fn: ', J)
        
    def forward_propagation (self, x_arr):
        # 1. 一行样本, 正向传播
        
        # layer = 2
        a1_1, a1_2, a2_1 = self.bias[0], self.bias[1], self.bias[2]
        for w, x in zip(self.weights[0:2], x_arr):
            a1_1 += w * x
        for w, x in zip(self.weights[2:4], x_arr):
            a1_2 += w * x

        a1 = sigmoid(np.array([a1_1, a1_2]))
        
        # layer = 3
        for w, x in zip(self.weights[4:6], a1):
            a2_1 += w * x

        a2_1 = sigmoid(a2_1)
        a2 = [a2_1]

        # x = [arr] 每次一个样本值
        # a1_1 = self.weights[0:2] * x_arr.T + self.bias[0]
        # a1_2 = self.weights[2:4] * x_arr.T + self.bias[1]
        # a2_1 = self.weights[4:6] * a1.T * self.bias[2]
        return [x_arr, a1, a2]
    
    def derivatives (self, a, y_true):
        a0, a1, a2 = a[0], a[1], a[2]
        a0_1, a0_2 = a0[0], a0[0]
        a1_1, a1_2 = a1[0], a1[1]
        a2_1 = a2[0]
        
        # w4, w5, b2
        dc_da21 = 2 * (a2_1 - y_true)
        da21_dz21 = derivative_sigmoid(a2_1)
        dz21_dw4 = a1_1
        dz21_dw5 = a1_2

        dc_dw4 = dc_da21 * da21_dz21 * dz21_dw4
        dc_dw5 = dc_da21 * da21_dz21 * dz21_dw5
        dc_db2 = dc_da21 * da21_dz21
        
        # w0, w1, b0
        dz21_da11 = self.weights[4] # w4
        dz21_da12 = self.weights[5] # w5

        dc_da11 = dc_da21 * da21_dz21 * dz21_da11
        dc_da12 = dc_da21 * da21_dz21 * dz21_da12
        
        da11_dz11 = derivative_sigmoid(a1_1)
        da12_dz12 = derivative_sigmoid(a1_2)
        
        dz11_dw0 = a0_1
        dz11_dw1 = a0_2
        dc_dw0 = dc_da11 * da11_dz11 * dz11_dw0
        dc_dw1 = dc_da11 * da11_dz11 * dz11_dw1
        dc_db0 = dc_da11 * da11_dz11
        
        # w2, w3, b1
        dz11_dw2 = a0_1
        dz11_dw3 = a0_2
        dc_dw2 = dc_da12 * da12_dz12 * dz11_dw2
        dc_dw3 = dc_da12 * da12_dz12 * dz11_dw3
        dc_db1 = dc_da12 * da12_dz12
        
        return dc_db0, dc_db1, dc_db2, dc_dw0, dc_dw1, dc_dw2, dc_dw3, dc_dw4, dc_dw5
    
    def gd (self, data):
        dc_db0, dc_db1, dc_db2, dc_dw0, dc_dw1, dc_dw2, dc_dw3, dc_dw4, dc_dw5 = data
        self.weights -= self.alpha * np.array([
            dc_dw0, dc_dw1, dc_dw2, dc_dw3, dc_dw4, dc_dw5
        ])
        
        self.bias -= self.alpha * np.array([
            dc_db0, dc_db1, dc_db2
        ])     
        
    def train (self, X, y):
        for i in range(self.loopnum):
            for x_arr, y_true in zip(X, y):
                # 一个样本, 返回所有激活值值
                a = self.forward_propagation(x_arr)
                # 反向推weights, bias导数
                data = self.derivatives(a, y_true)
                self.gd(data)
            
            if (i % 10 == 0):
                self.predict(X, y)

X = np.array([
    [-2, -1], # A
    [25, 6],  # B
    [17, 4],  # C
    [-15, -6],# D
])
y = np.array([
    1, # A
    0, # B
    0, # C
    1  # D
])

nn = NN()
nn.train(X, y)

Jessica = np.array([108-135, 60-66])
print('Jessica:')
print(nn.forward_propagation(Jessica))
# 0.9484256434415707 F

# 180, 80
Owen = np.array([180-135, 80-66])
print('Owen:')
print(nn.forward_propagation(Owen))
# 0.03935525212199268, M


"""
1. 有多少 layer?
2. 确定每个layer, 有多少个neurons?
3. 确定 weights, bias

4. 一次训练, 数据 X, y
   4.1 一个样本数据 X[i] y[i] (一行数据)
   4.2 正向传播: 最后一层 neurons eg: a21, a22 (第二层第一个, ...)
   4.3 和真实值差距: c = [a21, a22, ... a2n] - y
   4.4 求导数 (每个权重, 每个bias) dc / dw dc / dbias
   4.5 weights -= alpha * dc / dw
   4.6 bias -= alpha * dc / dbias
   4.7 i++, 循环4.1(直到X 所有行遍历完)
5. 循环1000次, 重复4
"""