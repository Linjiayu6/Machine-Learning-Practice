'''
Neuron: 
(1) a (activation): output value
(2) z (a = g(z)): input value
(3) g(x): sigmoid function
(4) w (weights): W = [w1, w2, w3 ...]
(5) b (bias)
(5) x (features = input): X = [x1, x2, x3 ...]

z = w1x1 + w2x2 + ... + wnxn + b
=> z = WX + b

a = g(z)
'''
import numpy as np

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
        
        # (2)matrix multiplication
        """
        W = np.matrix(self.weights)
        X = np.matrix(x)

        z = np.transpose(W) * X + self.bias
        a = sigmoid(z)
        """
        return a

# weights w1 = 0, w2 = 1
weights = np.array([
    [0],
    [1]
])
# bias = 4
bias = 4
n = Neuron(weights, bias)

# forward propagation features x1 = 2, x2 = 3
x = np.array([
    [2],
    [3]
])

print n.forward_propagation(x)
