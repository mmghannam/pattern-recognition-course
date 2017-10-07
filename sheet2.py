from numpy import matrix, transpose, dot, mean
from numpy.linalg import inv, eigh, det

# for formatting
print()

# 1-a
u1 = matrix([[3], [4], [0]])
u2 = matrix([[-4], [3], [0]])

dot_product = transpose(u1) * u2
# print(dot_product)  # == 0, then they are orthogonal

# 1-b
point = matrix([[6, 3, -2]])
projection_on_u1 = dot(u1, (dot(transpose(u1), transpose(point)) / dot(transpose(u1), u1)))
projection_on_u2 = dot(u2, (dot(transpose(u2), transpose(point)) / dot(transpose(u2), u2)))
# print(projection_on_u1)
# print(projection_on_u2)


# 2-a
class_1_instances = matrix([
    [4., 2.9],
    [3.5, 4.]
]).transpose()

class_2_instances = matrix([
    [2.5, 1.],
    [2., 2.1]
]).transpose()

mean1 = mean(class_1_instances, axis=1)
# print(mean1)

mean2 = mean(class_2_instances, axis=1)
# print(mean2)

difference_of_means = mean1 - mean2

between_class_scatter_matrix = difference_of_means * transpose(difference_of_means)
# print(between_class_scatter_matrix)


# 2-b

centered_1 = class_1_instances - mean1
centered_2 = class_2_instances - mean2

scatter_matrix_1 = centered_1 * transpose(centered_1)
scatter_matrix_2 = centered_2 * transpose(centered_2)

within_class_scatter_matrix = scatter_matrix_1 + scatter_matrix_2
# print(within_class_scatter_matrix)

# 2-c
eig_values, eig_vectors = eigh(inv(within_class_scatter_matrix) * between_class_scatter_matrix)
# print(eig_values) # the second value is higher then the second eigen vector better separates the classes

best_direction_to_separate = eig_vectors[1]
# print(best_direction_to_separate)


# 3-a

positive_class_variables = matrix([[2., 3.],
                                   [3., 3.],
                                   [3., 4.],
                                   [5., 8.],
                                   [7., 7.]]).transpose()

negative_class_variables = matrix([[5., 4.],
                                   [6., 5.],
                                   [7., 4.],
                                   [7., 5.],
                                   [8., 2.],
                                   [9., 4.]]).transpose()

mean_positive = mean(positive_class_variables, axis=1)
# print(mean_positive)

mean_negative = mean(negative_class_variables, axis=1)
# print(mean_negative)

difference_of_means = mean_positive - mean_negative
# print(difference_of_means)

between_class_scatter_matrix2 = difference_of_means * transpose(difference_of_means)
# print(between_class_scatter_matrix)


# 3-b

s_inverse = matrix([
    [0.056, -0.029],
    [-0.029, 0.052]
])

eig_values2, eig_vectors2 = eigh(s_inverse * between_class_scatter_matrix2)
# print(eig_values2) # the second value is higher then the second eigen vector better separates the classes

best_direction_to_separate2 = eig_vectors2[1]
print(best_direction_to_separate2)
