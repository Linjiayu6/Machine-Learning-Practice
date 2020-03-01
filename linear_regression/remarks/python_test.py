import numpy as np

A = [2, 1, 0]
"""
[[1]
 [2]
 [3]]
"""
A_mat = np.mat(A).T
print(np.mat(A).T)

B = [2, 3, 4]
B_mat = np.mat(B).T

"""
[[ 0]
 [-2]
 [-4]]
"""
# totalnum
m = A_mat.shape[0]
C_mat = (A_mat - B_mat)
# 0 + 4 + 16 = 20
sum_C_mat = C_mat.T * C_mat
print(float(sum_C_mat / (2 * m)))