# 2-a,b

from numpy import array
from helpers import k_means, scatter_plot_assignment

data = array([[0.5, 4.5, 2.5],
              [2.2, 1.5, 0.1],
              [3.9, 3.5, 1.1],
              [2.1, 1.9, 4.9],
              [0.5, 3.2, 1.2],
              [0.8, 4.3, 2.6],
              [2.7, 1.1, 3.1],
              [2.5, 3.5, 2.8],
              [2.8, 3.9, 1.5],
              [0.1, 4.1, 2.9]])

assignment = k_means(data, 3, 27)
# scatter_plot_assignment(assignment)


# 2-c
from helpers import manhattan_distance

assignment_manhattan = k_means(data, 3, 27, manhattan_distance)
# scatter_plot_assignment(assignment_manhattan)

# 2-d
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data)

assignment_normalized = k_means(normalized_data, 3, 1)
# scatter_plot_assignment(assignment_normalized)



