import numpy as np
import matplotlib.pyplot as plt

__author__ = "Wei-Chung Huang"
__copyright__ = "Copyright 2017"
__version__ = "1.0.0"


def my_lda(x, y):
    mean_x = np.mean(x, axis=1)
    mean_y = np.mean(y, axis=1)

    # within-class scatter
    sx = np.zeros([2, 2])
    for xi in x.T:
        xi = np.asmatrix(xi - mean_x).T * np.asmatrix(xi - mean_x)
        sx += xi

    sy = np.zeros([2, 2])
    for yi in y.T:
        yi = np.asmatrix(yi - mean_y).T * np.asmatrix(yi - mean_y)
        sy += yi

    sw = sx + sy

    # between-class scatter
    sb = np.asmatrix(mean_x - mean_y).T * np.asmatrix(mean_x - mean_y)

    sw_inv = np.linalg.inv(sw)

    s = sw_inv * sb
    eig_val, eig_vec = np.linalg.eig(s)

    large_eig_val_ind = np.argmax(eig_val)
    w = eig_vec[:, large_eig_val_ind]

    # project all data
    x_p = w.T * x
    y_p = w.T * y

    mean_x_p = np.mean(x_p)
    mean_y_p = np.mean(y_p)

    sx_p = 0
    for i in range(x_p.shape[1]):
        sx_p_i = x_p[0, i]
        sx_p += (sx_p_i - mean_x_p) ** 2

    sy_p = 0
    for i in range(y_p.shape[1]):
        sy_p_i = y_p[0, i]
        sy_p += (sy_p_i - mean_y_p) ** 2

    # Fisher discriminant
    m_w = (mean_x_p - mean_y_p) ** 2
    c_w = sx_p ** 2 + sy_p ** 2
    j_w = m_w / c_w
    return j_w


def swap(x, y, i, j):
    tmp = x[:, i].copy()
    x[:, i] = y[:, j]
    y[:, j] = tmp


def hill_climb(x, y):
    # hill-climb
    score_orig = my_lda(x, y)
    print(score_orig)
    for i in range(x.shape[1]):
        for j in range(y.shape[1]):
            x_new = x.copy()
            y_new = y.copy()
            swap(x_new, y_new, i, j)
            score = my_lda(x_new, y_new)
            print(i, j, score)
            if score > score_orig:
                return x_new, y_new

    return None


def test():
    x = [[1.25, 3], [1.5, 2], [2, 2.75], [2.25, 2], [2, 0.5], [3.25, 0.75], [3.5, 2.25], [4.25, 0.75]]
    y = [[2.75, 3.5], [3.25, 3], [4.5, 2.75], [3.5, 4.75]]
    # x = [[1.25, 3], [1.5, 2], [2, 2.75], [2.25, 2], [2.75, 3.5], [3.25, 3], [4.5, 2.75], [3.5, 4.75]]
    # y = [[2, 0.5], [3.25, 0.75], [3.5, 2.25], [4.25, 0.75]]
    x = np.asarray(x).transpose()
    y = np.asarray(y).transpose()

    plt.scatter(x[0, :], x[1, :])
    plt.scatter(y[0, :], y[1, :])
    plt.grid()
    plt.show()

    # hill-climb
    stop = False
    iteration = 0
    while not stop:
        iteration += 1
        print('iter:', iteration)
        new_xy = hill_climb(x, y)
        if new_xy is None:
            stop = True
        else:  # there is new pair improved
            x = new_xy[0].copy()
            y = new_xy[1].copy()

    print(x)
    print(y)
    plt.scatter(x[0, :], x[1, :])
    plt.scatter(y[0, :], y[1, :])
    plt.grid()
    plt.show()


if __name__ == '__main__':
    test()
