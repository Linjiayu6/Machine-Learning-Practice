
"""
http://lijiancheng0614.github.io/scikit-learn/modules/generated/sklearn.datasets.load_diabetes.html

Have problems !!!
"""
from sklearn import datasets
import numpy as np

# Datasets
def dataset():
    diabetes = datasets.load_diabetes()

    # ['target', 'DESCR', 'feature_names', 'target_filename', 'data', 'data_filename']
    # print(diabetes.keys())

    data = diabetes.data
    target = diabetes.target
    
    # https://github.com/Linjiayu6/Linear-Regression/wiki/Dataset
    X = data[:, :1] # y = a0 + a1x1 (only one feature x1 )
    Y = target
    
    # training set (first 80% data)
    """
    X = [  
        [ 0.03807591]
        [-0.00188202]
        ...
    ]
    Y = [233.  91. 111. ...] (After Matrix Transpose)
    """
    X_train = X[:-20]
    Y_train = Y[:-20].reshape((-1,1))
    # cross validation set
    # test set (last 20% data)
    X_test = X[-20:]
    Y_test = Y[-20:].reshape((-1,1))

    return X_train, Y_train, X_test, Y_test
    
class LinearRegression:
    # model: Y = WX + b
    def __init__(self):
        self.W = None
        self.b = None
    
    def cost_function(self, X, Y):
        print(self.W, self.b) # (array([[0.]]), 0)
        print(X, Y)
        # 1. Model SUM( (y_predict - y_train) ^ 2 ) / (2 * number)
        Hypothesis = X.dot(self.W) + self.b

        # 2. Cost function
        x_train_number = X.shape[0] # number of X data
        costVal = sum(np.square(Hypothesis - Y)) / (2 * x_train_number)

        # 3.1 derivative W: 2 * (y_predict_j - y_train) * x_j (only one)
        dH_dW = X.T.dot(Hypothesis - Y) / x_train_number
        
        # 3.2 derivative b: 2 * (y_predict_j - y_train) * 1 (only one)
        dH_db = np.sum(Hypothesis - Y) / x_train_number

        return costVal, dH_dW, dH_db
    
    def train(self, X, Y, learning_rate, loop_num):
        w_features_number = X.shape[1] # how many features have? = 1
        # [[0]] one feature and one weigts
        self.W = np.zeros((w_features_number, 1))
        self.b = 0
        costValue_list = []

        for i in range(loop_num):
            costVal, dH_dW, dH_db = self.cost_function(X, Y)
            # put costVal into costValue_list
            costValue_list.append(costVal)
            # gradient descent
            self.W -= learning_rate * dH_dW
            self.b -= learning_rate * dH_db
            
            if i % 500 == 0:
                # print('iters = %d, costfunction = %f' % (i, costVal))
                
        return costValue_list
    
    def predict(self, X_test):
        return X_test.dot(self.W) + self.b

# 1. dataset
X_train, Y_train, X_test, Y_test = dataset()

# 2. train
model = LinearRegression()
costValue_list = model.train(X_train, Y_train, 0.01, 1000)
print(model.W, model.b)

# 3. predict
model.predict(X_test)

# 4. data visualization
import matplotlib.pyplot as plt

f = X_train.dot(model.W) + model.b
fig = plt.figure()
plt.subplot(211)
plt.scatter(X_train, Y_train,color = 'black')
plt.scatter(X_test, Y_test,color = 'blue')
plt.plot(X_train,f,color = 'red')
plt.xlabel('X')
plt.ylabel('y')
 
plt.subplot(212)
plt.plot(costValue_list, color = 'blue')
plt.xlabel('epochs')
plt.ylabel('errors')
plt.show()