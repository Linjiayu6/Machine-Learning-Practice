# -*- coding:utf-8 -*-

import numpy as np
"""
Neuron

1. layer_i: 第i层
2. node_i: 第i层的第j个节点
3. z: xxx
4. a: 激活值(activiation)
"""

def sigmoid (z):
    return 1 / (1 + np.exp(-z))

class Neuron ():
    def __init__ (self, layer_i, node_i):
        self.layer_i = layer_i
        self.node_i = node_i

        self.z = 0
        self.a = 0
    
    def a (self):
        return self.a

    def cal_activiation (self, inputs_arr, weights_arr, bias):
        # inputs_arr = [1, 2, 3]
        # weights_arr = [1, 1, 1] bias = 1
        z = inputs_arr * np.mat(weights_arr).T + bias
        # z = [[ 1 ]] np.array(z)[0][0] = 1
        z = np.array(z)[0][0]
        self.z = z
        self.a = sigmoid(z)

        # self.print_()
        return self.a
    
    def print_ (self):
        print('self.layer_i', self.layer_i)
        print('self.node_i', self.node_i)
        print('z:', self.z)
        print('a', self.a)

class Layer ():
    # 输入值, 第几层, 该层有n个结点
    def __init__ (self, inputs_arr, layer_i, n):
        # 输入层
        self.inputs_arr = inputs_arr
        features_len = len(inputs_arr)
        # 第几层
        self.layer_i = layer_i
        # 第几层有 n个结点 (例如2个结点)
        self.n = n
        # 权重
        self.weights_mat = np.random.rand(self.n, features_len)
        # bias [1, 1]
        self.bias = np.random.rand(self.n)
        # 该层, 所有激活值
        self.a_arr = []

        self.forward_propagation()
        
    def data (self):
        return self.a_arr, self.weights_mat, self.bias

    def forward_propagation (self):
        # 该层如果有2个结点
        for i in range(self.n):
            n = Neuron(self.layer_i, i)
            a = n.cal_activiation(self.inputs_arr, self.weights_mat[i], self.bias[i])
            self.a_arr.append(a)

        # return self.weights_mat, self.bias, self.a_arr

# # 第一层
# first_arr = [1, 2, 3]
# # 第二层 (第2层, 有3个结点)
# second_weights, second_bias, second_arr = Layer(first_arr, 1, 3).forward_propagation()

# # 第三层 (第3层, 有2个结点)
# third_weights, third_bias, third_arr = Layer(second_arr, 2, 2).forward_propagation()

# A.append(first_arr)
# A.append(second_arr)
# A.append(third_arr)
# print(A)

# Weights.append(second_weights)
# Weights.append(third_weights)
# print(Weights)

# Bias.append(second_bias)
# Bias.append(third_bias)
# print(Bias)

A = [ [1, 2, 3] ]
Weights = []
Bias = []
L = []

# hidden层 或 输出层 每层的节点数
nodes = np.array([0, 3, 2])

# hidden 第一层, hidden 第二层(output层)
for i in range(1, 3):
    # 上一个层数据, 第几层, 当前层数有几个结点
    layer = Layer(A[-1], i, nodes[i])
    L.append(layer)

    a_arr, weights_mat, bias = layer.data()

    A.append(a_arr)
    Weights.append(weights_mat)
    Bias.append(bias)

print(L)
print(A)
print(Weights)
print(Bias)