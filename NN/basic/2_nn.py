
# Create a network

import numpy as np

# ======== 1_Neuron.py ========
# sigmoid function
def sigmoid(z):
    # output between 0 and 1
    # g(z) = 1 / 1 + e^-z
    return 1 / (1 + np.exp(-z))

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def forward_propagation(self, x):
        '''
        z = W * X + b
        a = g(z)
        '''
        # (1) dot product (np.vdot for vector dot product)
        z = np.vdot(self.weights, x) + self.bias
        a = sigmoid(z)
        
        # (2)matrix multiplication
        """
        W = np.matrix(self.weights)
        X = np.matrix(x)

        z = np.transpose(W) * X + self.bias
        a = sigmoid(z)
        """
        return a
# ======== END ========

"""
1. input layer: x1, x2
2. hidden layer: h1, h2
3. output layer: o1

all weights = [0 1] w1 = 0 w2 = 1
all bias = 0

statements:
h1 = w1x1 + w2x2 + b
"""
class Simple_NN:
    def __init__(self):
        self.weights = np.array([0, 1]) # w1 = 0, w2 = 0
        self.bias = 0
        
    def forward_propagation(self, Inputs):
        h1 = Neuron(self.weights, self.bias).forward_propagation(Inputs)
        h2 = Neuron(self.weights, self.bias).forward_propagation(Inputs)
        
        secondlayer_inputs = np.array([h1, h2])
        o1 = Neuron(self.weights, self.bias).forward_propagation(secondlayer_inputs)
        
        print h1, h2
        return o1

Features = np.array([2, 3])
print Simple_NN().forward_propagation(Features)
# 0.721632560952