from numpy.linalg import norm
from numpy import array as np_array, transpose, dot, mean
from math import sqrt
from pprint import pprint

# to easily capture results
print()

data = np_array([[10., 60., 10., 90.],
                 [20., 50., 40., 70.],
                 [30., 50., 30., 40.],
                 [20., 50., 20., 60.],
                 [10., 60., 30., 10.]])
# 1-c
distance1_3 = norm(data[0] - data[2])
# print(distance1_3)

# 1-d
length_x2 = norm(data[1])
# print(length_x2)

# 1-e
a = data[1]
b = data[3]
numerator = dot(transpose(a), b)
denominator = norm(a) * norm(b)
# print(numerator / denominator)

# 1-g
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data)
# print(data)

# 1-h
new_distance1_3 = norm(normalized_data[0] - normalized_data[2])
# print(distance1_3)

new_length_x2 = norm(normalized_data[1])
# print(length_x2)

a = normalized_data[1]
b = normalized_data[3]
new_numerator = dot(transpose(a), b)
new_denominator = norm(a) * norm(b)
# print(numerator / denominator)

# 2-a
# print(transpose(norm(data, axis=1)))

# 2-b
from sklearn.metrics.pairwise import cosine_similarity
# print(cosine_similarity(data))

# 2-c
from sklearn.metrics.pairwise import euclidean_distances

# print(euclidean_distances(data))

# 3-a
new_data = (data[:, [0, 1, 3]])
# print(new_data)

# 3-b
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
axis = Axes3D(fig)

axis.scatter(*transpose(new_data))

# plt.show()


# 3-c
from numpy import mean, newaxis

mean_vector = mean(new_data, axis=0)
# print(mean_vector)

# 3-d
centered_data_matrix = new_data - mean_vector
# print(centered_data_matrix)

from numpy import cov

print((1. / len(new_data)) * dot(transpose(new_data), new_data))
