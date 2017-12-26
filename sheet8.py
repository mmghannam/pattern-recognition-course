from numpy import array

data_A = array([[2, 1],
                [2, 2],
                [2, 3],
                [2, 4],
                [3, 2],
                [3, 3],
                [4, 3],
                [5, 1],
                [5, 2],
                [5, 3],
                [5, 4],
                [6, 3],
                [6, 4],
                [6, 5],
                [7, 2],
                [8, 5],
                [10, 1],
                [10, 3],
                [10, 5],
                [11, 3],
                [11, 4],
                [12, 2],
                [13, 5]
                ])

labels = [2, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 3, 1, 3, 3, 3, 3, 3, 3, 3]

# from sklearn import svm
#
# clf = svm.SVC()
# clf.fit(data_A, labels, kernel="rbf")



import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


X = data_A
y = [int(x == 3) for x in labels]

Cs = [0.1, 0.3, 1.0, 1, 3, 10]
models = []
for C in Cs:
    models.append(svm.LinearSVC(C=C))


models = (clf.fit(X, y) for clf in models)

# print([model.support_vectors_ for model in models])

titles = ('C = 0.1',
          'C = 0.3',
          'C = 1',
          'C = 3',
          'C = 10')

fig, sub = plt.subplots(2, 3)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()

data_B = array([[1, 5],
                [2, 6],
                [2, 10],
                [2, 12],
                [3, 17],
                [3, 12],
                [4, 6],
                [4, 5],
                [4, 7],
                [5, 10]
                ])

labels_B = [10, 40, 50, 60, 70, 50, 30, 20, 40, 70]
