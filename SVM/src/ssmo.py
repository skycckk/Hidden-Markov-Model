import numpy as np
import random
import matplotlib.pyplot as plt


def plot_hyperlane(L, X, z, b):
    """
    Plot sample points and separate line in one graph.
    :param L: ndarray
            Lambda elements with size of 1xN array
    :param X: ndarray
            MxN array where M is sample size and N is feature length
    :param z: ndarray
            Labels with size of 1xN array
    :param b: float
            B value of SVM scoring function
    :return:
    """
    num_samples = X.shape[0]
    coeff_x, coeff_y = 0, 0
    for i in range(num_samples):
        if abs(L[i] - 0) > 10e-7:
            coeff_x += L[i] * z[i] * X[i, 0]
            coeff_y += L[i] * z[i] * X[i, 1]

    coeff_A = -coeff_x / coeff_y
    coeff_C = -b / coeff_y
    equation = str(coeff_A) + '*x+' + str(coeff_C)
    print(equation)
    graph(equation, range(-10, 11), X, z)


def graph(formula1, x_range, X, z, formula2=None):
    """
    Plot a line by taking xy formula in 2D space.
    Also, plot the sample points.
    :param formula1: string
            Formula X and Y in the form of y = ax + b, that is: '123*x+456'
    :param x_range:
    :param X: ndarray
            MxN array where M is sample size and N is feature length
    :param z: ndarray
            Labels with size of 1xN array
    :param formula2: string (optional)
            Another line needs to be plot
    :return:
    """
    x = np.array(x_range)
    y = eval(formula1)
    axes = plt.gca()
    axes.set_xlim([0, 5])
    axes.set_ylim([0, 5])
    plt.plot(x, y, label="svm-result 1")

    if formula2 is not None:
        plt.plot(x, eval(formula2), 'g--', label="svm-result 2")

    for i in range(X.shape[0]):
        if (z[i] == -1):
            plt.plot(X[i, 0], X[i, 1], marker='o', markersize=5, color='red')
        else:
            plt.plot(X[i, 0], X[i, 1], marker='s', markersize=5, color='blue')
    plt.legend()
    plt.show()


def score(X, L, z, b, y):
    """
    Score SVM by using linear kernel
    :param X: ndarray
            MxN array where M is sample size and N is feature length
    :param L: ndarray
            Lambda elements with size of 1xN array
    :param z: ndarray
            Labels with size of 1xN array
    :param b: float
            B value of SVM scoring function
    :param y: ndarray
            Testing feature vector with size 1xN
    :return: float
            Score of applying f(x) <- hyperplane function
    """
    num_samples = X.shape[0]
    sum = 0
    for i in range(num_samples):
        sum += L[i] * z[i] * X[i].dot(y)
    sum += b

    return sum


def get_pair_list_rand(n):
    """
    By using random approach to get list of (i, j) pairs where i is not equal j.
    :param n: int
            Size of samples
    :return: int
            list of (i, j) pairs where (i, j) is tuple
    """
    pairs = list()
    rem = 1000
    for start_index in range(n):
        curr = min(rem, int(1000 / n))
        for j in range(curr):
            allow_numbers = list(range(0, start_index)) + list(range(start_index + 1, n))
            end_index = random.choice(allow_numbers)
            pairs.append((start_index, end_index))

        rem = rem - curr

    # THIS FAIL
    # for i in range(1000):
    #     allow_numbers = list(range(0, start_index)) + list(range(start_index + 1, n))
    #     end_index = random.choice(allow_numbers)
    #
    #     pairs.append((start_index, end_index))
    #     start_index = (start_index + 1) % n

    return pairs


def get_pair_list(n):
    """
    By using hill-climbing approach to get list of (i, j) pairs where i is not equal j.
    :param n: int
            Size of samples
    :return: int
            list of (i, j) pairs where (i, j) is tuple
    """
    pairs = list()
    for interval in range(1, n):
        i = 0
        j = i + interval
        while j < n:
            pairs.append((i, j))
            i += 1
            j += 1

    num_pairs = len(pairs)
    for i in range(num_pairs):
        pair = pairs[i]
        pairs.append((pair[1], pair[0]))

    return pairs


def ssmo(X, z, C, epislon, visualization=False):
    """
    Simple version of SMO algorithm (Sequential Minimal Optimization).
    This is based on a linear kernel and it is used to solve the quadratic programming
    in SVM.
    That is, the lambda and b value solved by this function can generate a hyberplane that
    separate X.
    :param X: ndarray
            MxN array where M is sample size and N is feature length
    :param z: ndarray
            Labels with size of 1xN array
    :param C: float
            Regularization parameter.
    :param epislon: float
            Numerical tolerance.
    :param visualization: bool
            A flag to plot the hyperplane in each iteration.
    :return:
    """
    num_samples = X.shape[0]

    # initialize L, b
    L = np.zeros(num_samples, dtype=np.float32)
    b = 0

    # body
    num_passes = 0
    max_iter = 100
    # select i, j pair
    pairs = get_pair_list_rand(num_samples)
    while num_passes < max_iter:
        print('pass', num_passes)

        L_old = L.copy()
        for pair in pairs:
            i = pair[0]
            j = pair[1]

            Xi, Xj = X[i], X[j]
            Li, Lj = L[i], L[j]
            zi, zj = z[i], z[j]
            d = 2 * Xi.dot(Xj.T) - Xi.dot(Xi.T) - Xj.dot(Xj.T)
            if np.abs(d) > epislon:
                Ei = score(X, L, z, b, Xi) - zi
                Ej = score(X, L, z, b, Xj) - zj

                Li_old, Lj_old = Li, Lj
                Lj = Lj - (zj * (Ei - Ej) / d)

                l, h = 0, 0
                if zi != zj:
                    l = max(0, Lj - Li)
                    h = min(C, C + Lj - Li)
                else:
                    l = max(0, Li + Lj - C)
                    h = min(C, Li + Lj)

                if Lj > h: Lj = h
                elif Lj < l: Lj = l

                Li = Li + zi * zj * (Lj_old - Lj)
                bi = b - Ei - zi * (Li - Li_old) * Xi.dot(Xi.T) - zj * (Lj - Lj_old) * Xi.dot(Xj.T)
                bj = b - Ej - zi * (Li - Li_old) * Xi.dot(Xj.T) - zj * (Lj - Lj_old) * Xj.dot(Xj.T)

                if C > Li > 0:
                    b = bi
                elif C > Lj > 0:
                    b = bj
                else:
                    b = (bi + bj) / 2

                L[i], L[j] = Li, Lj

        if visualization:
            plot_hyperlane(L, X, z, b)

        if np.abs(np.sum(L - L_old)) < 10e-7:
            num_passes = max_iter
        else:
            num_passes += 1

    return L, b


if __name__ == '__main__':
    X = np.zeros([6, 2], dtype=np.float32)
    X[0] = 3, 3
    X[1] = 3, 4
    X[2] = 2, 3
    X[3] = 1, 1
    X[4] = 1, 3
    X[5] = 2, 2

    z = np.zeros(6, dtype=np.float32)
    z[0] = 1
    z[1] = 1
    z[2] = 1
    z[3] = -1
    z[4] = -1
    z[5] = -1

    random.seed(1)
    L, b = ssmo(X, z, 2.5, 0.00001, visualization=False)
    print('lambda', L)
    print('b', b)


    # Formula of using non-random pair
    # -0.621520729296*x+3.89164975031
    # -0.608370713546*x+3.86008971251

    # Formula of using random pair
    # -0.999999618531*x+4.79999816895
    # graph('-0.608370713546*x+3.86008971251', range(-10, 11), X, z, formula2='-0.999999618531*x+4.79999816895')