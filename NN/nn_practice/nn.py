# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def read_data ():
    # MNIST 是由美国的高中生和美国人口调查局的职员手写数字（0-9）图片
    data_file = open("datasets/training_set.csv", 'r')
    data_array = data_file.readlines()
    data_file.close()

    # 100 samples, 一行数据一共有 785 个数据, 第一个值为真实数字, 后为784个像素值(28*28)
    item_arr = data_array[10].split(',')
    # image_array = (np.asfarray(item_arr[1:]) / 255 * 9).astype(int).reshape(28, 28)
    image_array = np.array([int(float(item_arr[i]) / 255 * 9)  for i in range(len(item_arr[1:]))])
    image_array = image_array.reshape(28, 28)    
    
    # plt.imshow(image_array, cmap='Greys', interpolation='None')
    # plt.show()
# read_data()

class NerualNetwork():
    def __init__ (self, input_num, hidden_num, output_num, alpha):
        # 设置 input layer, hidden layer, output layer neuron的数量
        self.input_num = input_num
        self.hidden_num = hidden_num
        self.output_num = output_num
        
        # weights: 从输入层到隐藏层, 从隐藏层到输出层
        self.input_hidden_weights = np.random.rand(hidden_num, input_num) - 0.5
        self.hidden_output_weights = np.random.rand(output_num, hidden_num) - 0.5

        # learning rate
        self.alpha = alpha
        
        # sigmoid function
        self.sigmoid_fn = lambda z: 1 / (1 + np.exp(-z))
        
        # sigmoid derivatives
        self.sigmoid_derivatives = lambda activiation: activiation * (1 - activiation)
        pass
    
    def forward_propagation (self, inputs_list):
        # 1. 将输入的数组转化为一个二维数组
        inputs = np.array(inputs_list, ndmin=2).T
        # [1,2,3] => [[1,2,3]].T
        
        # 2. hidden layers (z = hidden_inputs; a = sigmoid(z) = hidden_outputs)
        hidden_inputs = np.dot(self.input_hidden_weights, inputs)
        hidden_outputs = self.sigmoid_fn(hidden_inputs)
        
        # 3. output layers
        final_inputs = np.dot(self.hidden_output_weights, hidden_outputs)
        final_outputs = self.sigmoid_fn(final_inputs)

        return final_outputs
    
    def train (self, inputs_list, target_list):
        # 1. 输入值 和 目标值 处理
        inputs = np.array(inputs_list, ndmin = 2).T
        targets = np.array(target_list, ndmin = 2).T


        # 2. 向前传播
        # 隐藏层输入输出
        hidden_inputs = np.dot(self.input_hidden_weights, inputs)
        hidden_outputs = self.sigmoid_fn(hidden_inputs)
        # 最后层输入输出
        final_inputs = np.dot(self.hidden_output_weights, hidden_outputs)
        final_outputs = self.sigmoid_fn(final_inputs)


        # 3. 差值: 预测差值, 隐藏层的差值
        print(targets, final_outputs)
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.hidden_output_weights.T, output_errors)


        # 4. 向后传播
        self.hidden_output_weights -= self.alpha * np.dot(
            output_errors * self.sigmoid_derivatives(final_outputs),
            hidden_outputs.T
        )
        
        self.input_hidden_weights -= self.alpha * np.dot(
            hidden_errors * self.sigmoid_derivatives(hidden_outputs),
            inputs.T
        )
        
        print(self.hidden_output_weights, self.input_hidden_weights)
        
        pass
    
    def predict (self):
        pass

NN = NerualNetwork(4, 3, 1, 0.1)
NN.train([1, 2, 6, 5], [0])