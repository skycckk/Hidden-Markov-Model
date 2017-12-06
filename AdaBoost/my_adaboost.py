import numpy as np

__author__ = "Wei-Chung Huang"
__copyright__ = "Copyright 2017"
__version__ = "1.0.0"


def my_adaboost():
    x =  [ 1, -1, -1,  1,  1,  1,  1, -1,  1, -1]
    c1 = [-1, -1, -1, -1, -1, -1, -1, -1,  1,  1]
    c2 = [-1, -1, -1, -1,  1,  1, -1,  1,  1, -1]
    c3 = [-1,  1,  1, -1,  1, -1,  1,  1, -1,  1]
    c4 = [ 1,  1, -1,  1,  1,  1, -1, -1,  1, -1]
    c5 = [ 1,  1, -1,  1, -1, -1,  1, -1, -1,  1]
    n = len(x)
    classifiers = [c1, c2, c3, c4, c5]

    unused_c = [c1, c2, c3, c4, c5]
    a_list = list()
    k_list = list()
    for k in range(len(classifiers)):
        # pick one with min miss
        new_W2 = np.finfo(np.float32).max
        new_W1 = new_W2
        selected_index_c = 0
        selected_class = None
        for index_c in range(len(unused_c)):
            c = unused_c[index_c]
            W1, W2 = 0, 0
            for i in range(n):
                zi = x[i]
                if c[i] != zi:  # miss count
                    W2 += np.exp(-zi * classifier_m(k - 1, i, a_list, k_list))
                else:
                    W1 += np.exp(-zi * classifier_m(k - 1, i, a_list, k_list))

            if W2 <= new_W2:  # pick this weak classifier
                selected_index_c = index_c
                selected_class = c.copy()
                new_W2 = W2
                new_W1 = W1

        # remove selected c
        unused_c.pop(selected_index_c)

        # decide alpha
        new_W = new_W1 + new_W2
        rm = new_W2 / new_W
        print(rm)
        am = np.log((1 - rm) / rm) / 2.0
        a_list.append(am)
        k_list.append(selected_class)
        print(a_list, '*' * 10)

        hit = 0
        for i in range(n):
            r = classifier_m(k, i, a_list, k_list)
            if x[i] * r > 0:
                hit += 1
        print(hit / n * 100)


def classifier_m(stage, xi, a_list, k_list):
    if stage < 0:
        return 0

    return classifier_m(stage - 1, xi, a_list, k_list) + a_list[stage] * k_list[stage][xi]


if __name__ == '__main__':
    my_adaboost()
