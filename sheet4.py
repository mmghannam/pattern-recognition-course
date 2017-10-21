from numpy import array
from numpy.linalg import norm

data_matrix = array([[2, 4],
                     [3, 3],
                     [3, 4],
                     [5, 4],
                     [5, 6],
                     [5, 8],
                     [6, 4],
                     [6, 5],
                     [6, 7],
                     [7, 3],
                     [7, 4],
                     [8, 2],
                     [9, 4],
                     [10, 6],
                     [10, 7],
                     [10, 8],
                     [11, 5],
                     [11, 8],
                     [12, 7],
                     [13, 6],
                     [13, 7],
                     [14, 6],
                     [15, 4],
                     [15, 5]])

from sklearn.metrics.pairwise import rbf_kernel as rbf

rbf_001 = rbf(data_matrix, None, 0.01)
rbf_01 = rbf(data_matrix, None, 0.1)
rbf_1 = rbf(data_matrix, None, 1)
rbf_10 = rbf(data_matrix, None, 10)

import matplotlib.pyplot as plt

# plt.matshow(rbf_001)
# plt.matshow(rbf_01)
# plt.matshow(rbf_1)
# plt.matshow(rbf_10)
# plt.show()

from networkx import laplacian_matrix, from_numpy_array
lp_matrix = laplacian_matrix(from_numpy_array(rbf_001)).to_array
# print(lp_matrix)
