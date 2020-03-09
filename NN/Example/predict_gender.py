
import numpy as np

def sigmoid(z):
    # 1 / 1 + e^(-z)
    return 1 / (1 + np.exp(-z))

def derivative_sigmoid(z):
    # derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(z)
    return fx * (1 - fx)

def cost_function(Y_true, Y_predict):
    # sum_{i=1}^n ((Y_predict - Y_true) ^ 2) / n
    return ((Y_true - Y_predict) ** 2).mean()

def derivative_cost_function(y_true, y_predict):
    # dC / dy_predict
    return -2*(y_true - y_predict)
    
# activation, weights, bias
class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    
    def create(self, inputs):
        # z = (w1x1 + w2x2 + ... +wnxn) + bias
        # z = W dot X = W^(T) X + bias
        # a = sigmoid(z)

        #(1) dot prouct or (2) matrix multiplication
        z = np.vdot(self.weights, self.inputs) + bias
        return sigmoid(z)

class NeuralNetwork:
    def __init__(self):
        # create randomly weights and bias
        # (w1, w2, b1) => h1 
        # (w3, w4, b2) => h2
        # (h1, w5, w6, b3) => o1
        
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
        
    # features: X = [x1, x2]
    def forward_propagation(self, X):
        # ([x1, x2], w1, w2, b1) => h1 
        h1 = sigmoid(self.w1 * X[0] + self.w2 * X[1] + self.b1)
        # ([x1, x2], w3, w4, b2) => h2
        h2 = sigmoid(self.w3 * X[0] + self.w4 * X[1] + self.b2)
        # ([h1, h2],h1, w5, w6, b3) => o1
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1
    
    # aim: minimize the cost function
    # partial derivatives of all weights and bias
    def step1_activation_z(self, X_features, y_true):
        # X_features: all features
        # y_true: true result
        """
        All Data
        """
        # ([x1, x2], w1, w2, b1) => h1 
        z_h1 = self.w1 * X_features[0] + self.w2 * X_features[1] + self.b1
        h1 = sigmoid(z_h1)
        # ([x1, x2], w3, w4, b2) => h2
        z_h2 = self.w3 * X_features[0] + self.w4 * X_features[1] + self.b2
        h2 = sigmoid(z_h2)
        # ([h1, h2],h1, w5, w6, b3) => o1
        z_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
        o1 = sigmoid(z_o1)

        return z_h1, h1, z_h2, h2, z_o1, o1
        
    def step2_derivatives_w_b(self, neurons, X_features, y_true):
        z_h1, h1, z_h2, h2, z_o1, o1 = neurons

        """
        Cost function => C(w1,w2,b1,w3,w4,b2,w5,w6,b3)
        """
        y_predict = o1
        C = cost_function(y_true, y_predict)
        dC_dypredict = derivative_cost_function(y_true, y_predict)
        
        # Weights:

        # dC_dw1 = dC_do1 * do1_dh1 * dh1_dw1
        dC_do1 = dC_dypredict
        do1_dh1 = derivative_sigmoid(z_o1) * self.w5
        dh1_dw1 = derivative_sigmoid(z_h1) * X_features[0]
        
        # dC_dw2 = dC_do1 * do1_dh1 * dh1_dw2
        dh1_dw2 = derivative_sigmoid(z_h1) * X_features[1]
        
        # dC_dw3 = dC_do1 * do1_dh2 * dh2_dw3
        do1_dh2 = derivative_sigmoid(z_o1) * self.w6
        dh2_dw3 = derivative_sigmoid(z_h2) * X_features[0]
        
        # dC_dw4 = dC_do1 * do1_dh2 * dh2_dw4
        dh2_dw4 = derivative_sigmoid(z_h2) * X_features[1]
        
        # dC_dw5 = dC_do1 * do1_dw5
        do1_dw5 = derivative_sigmoid(z_o1) * h1
        
        # dC_dw6 = dC_do1 * do1_dw6
        do1_dw6 = derivative_sigmoid(z_o1) * h2
        
        dC_dw1 = dC_do1 * do1_dh1 * dh1_dw1
        dC_dw2 = dC_do1 * do1_dh1 * dh1_dw2
        dC_dw3 = dC_do1 * do1_dh2 * dh2_dw3
        dC_dw4 = dC_do1 * do1_dh2 * dh2_dw4
        dC_dw5 = dC_do1 * do1_dw5
        dC_dw6 = dC_do1 * do1_dw6
        
        # Bias:
        dh1_db1 = derivative_sigmoid(z_h1) * 1
        dh2_db2 = derivative_sigmoid(z_h2) * 1
        do1_db3 = derivative_sigmoid(z_o1) * 1

        # dC_db1 = dC_do1 * do1_dh1 * dh1_db1
        dC_db1 = dC_do1 * do1_dh1 * dh1_db1
        dC_db2 = dC_do1 * do1_dh2 * dh2_db2
        dC_db3 = dC_do1 * do1_db3
        
        return dC_dw1, dC_dw2, dC_dw3, dC_dw4, dC_dw5, dC_dw6, dC_db1, dC_db2, dC_db3
    
    def step3_gradientdescent(self, derivatives, learning_rate):
        dC_dw1, dC_dw2, dC_dw3, dC_dw4, dC_dw5, dC_dw6, dC_db1, dC_db2, dC_db3 = derivatives
        # # weights
        self.w1 -= learning_rate * dC_dw1
        self.w2 -= learning_rate * dC_dw2
        self.w3 -= learning_rate * dC_dw3
        self.w4 -= learning_rate * dC_dw4
        self.w5 -= learning_rate * dC_dw5
        self.w6 -= learning_rate * dC_dw6
        
        # # bias
        self.b1 -= learning_rate * dC_db1
        self.b2 -= learning_rate * dC_db2
        self.b3 -= learning_rate * dC_db3

    def training(self, X, Y):
        learning_rate = 0.1
        loop_num = 4000
        
        for _num in range(loop_num):
            """
            nums = ['a1','b2','c3']
            for i in zip(*nums):
                print(i)
            output: (a, b, c)
            output: (1, 2, 3)
            """
            for X_features, y_true in zip(X, Y):
                # 1. calculate the value of z and a(activation)
                neurons = self.step1_activation_z(X_features, y_true)

                # 2. calculate derivatives of all dC / dw1, dC / dw2, ...., dC / db1, dC / db2, ...
                derivatives = self.step2_derivatives_w_b(neurons, X_features, y_true)

                # 3. update all weights and bias
                self.step3_gradientdescent(derivatives, learning_rate)
                
                if _num % 10 == 0:
                    # o1 = y_predict
                    # apply_along_axis(func, axis, inputs)
                    Y_predict = np.apply_along_axis(self.forward_propagation, 1, X)
                    L = cost_function(Y, Y_predict)              
                    print("No %d costfn: %.6f" % (_num, L))

"""
(-2, -1) 1
(25, 6) 0
(17, 4) 0
(-15, -6) 1
"""
X = np.array([
    [-2, -1], # A
    [25, 6],  # B
    [17, 4],  # C
    [-15, -6],# D
])
Y = np.array([
    1, # A
    0, # B
    0, # C
    1  # D
])
# Training
nn = NeuralNetwork()
nn.training(X, Y)


# Make predictions
"""
> Weight: -135, Height: -66
> F: 1, M: 0
"""
# 108, 60
Jessica = np.array([108-135, 60-66])
print 'Jessica:'
print nn.forward_propagation(Jessica)
# 0.984217578557, F

# 180, 80
Owen = np.array([180-135, 80-66])
print 'Owen:'
print nn.forward_propagation(Owen)
# 0.0264516468082, M