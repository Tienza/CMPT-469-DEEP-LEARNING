import numpy as np

A = np.matrix('1 4 -3;2 -1 3')
B = np.matrix('-2 0 5;0 -1 4')

result = np.matmul(A.T, B)

print(result)
print("Rank:" + str(np.linalg.matrix_rank(result)))