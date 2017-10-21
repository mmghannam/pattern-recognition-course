from numpy.linalg import norm
from numpy import array,matrix, transpose, dot, diag, mean

data = matrix('3.5  1.4 0.2;'
              '3    1.4 0.2;'
              '3.2  1.3 0.2;'
              '3.1  1.5 0.2;'
              '3.2  4.7 1.4;'
              '3.2  4.5 1.5;'
              '3.1  4.9 1.5;'
              '2.3  4   1.3;'
              '2.8  5.6 2.1;'
              '3    5.8 1.6;'
              '2.8  6.1 1.9;'
              '3.8  6.4 2')

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
axis = Axes3D(fig)
axis.scatter(*array(data.T))

# plt.show()

mean_vector = mean(data, axis=0)
# print(mean_vector)

centered_data_matrix = data - mean_vector
# print(centered_data_matrix)

from numpy import cov

cov = (1. / len(centered_data_matrix)) * dot(transpose(centered_data_matrix), centered_data_matrix)
# print(cov)

from numpy.linalg import eigh

eig_values, eig_vectors = eigh(cov)
eig_values_diag = diag(eig_values)
# print(eig_vectors)
# print(diag(eig_values))

prefered_trans_matrix = eig_vectors[:, 1:]
# print(prefered_trans_matrix)

projected_data = data * prefered_trans_matrix
# print(projected_data)

fig2 = plt.figure()
plt.scatter(*array(projected_data.T),c=[0,0,0,0,1,1,1,1,2,2,2,2])
plt.show()
