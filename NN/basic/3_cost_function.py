
# Cost Function
# (y_predict - y_train) ^ 2)

import numpy as np

def cost_fn(Y_predict, Y_train):
    return ((Y_predict - Y_train) ** 2).mean()

Y_predict = np.array([1, 0, 0, 1])
Y_train = np.array([0, 0, 0, 0])

# (1-0 + 0-0 + 0-0 + 1-0) / 4 = 0.5

print cost_fn(Y_predict, Y_train)
