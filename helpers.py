from numpy import mean, array
from numpy.linalg import norm


def euclidean_distance(a, b):
    return norm(a - b)


def manhattan_distance(a, b):
    return sum(abs(xn - yn) for xn, yn in zip(a, b))


def k_means(data, k, error, distance_func=euclidean_distance, max_iter=100):
    import random
    t = 0
    assignment = {}
    for _ in range(k):
        assignment[tuple(random.choice(data))] = []

    while True:
        t += 1
        # cluster assignment step
        for sample in data:
            closest_mean = min(assignment.keys(), key=lambda x: distance_func(array(x), sample))
            assignment[closest_mean].append(sample)
        # centroid update step
        for last_mean in [key for key in assignment.keys()]:
            new_mean = mean(assignment[last_mean], axis=0)
            assignment[tuple(new_mean)] = assignment[last_mean]
            del assignment[last_mean]

        if sum_square_error(assignment) < error or t > max_iter:
            break
    return assignment


def sum_square_error(assignment):
    error = 0
    for mean in assignment.keys():
        for element in assignment[mean]:
            error += norm(element - mean) ** 2
    # print(error)
    return error


def scatter_plot_assignment(assignment):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure()
    axis = Axes3D(fig)
    colors = ['red', 'blue', 'green']
    color_index = 0
    for centroid in assignment.keys():
        axis.scatter(*assignment[centroid], color=colors[color_index])
        color_index += 1

    plt.show()



