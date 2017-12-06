import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance

__author__ = "Wei-Chung Huang"
__copyright__ = "Copyright 2017"
__version__ = "1.0.0"


def my_knn():
    red_pts = np.asarray([[0.5, 3.0], [1.0, 4.25], [1.5, 2.0], [2.0, 2.75],
                          [2.5, 1.65], [3.0, 2.7], [3.5, 1.0], [4.0, 2.5],
                          [4.5, 2.1], [5.0, 2.75]])
    blue_pts = np.asarray([[0.5, 1.75], [1.5, 1.5], [2.5, 4.0], [2.5, 2.1],
                           [3.0, 1.5], [3.5, 1.85], [4.0, 3.5], [5.0, 1.45]])

    for xi in range(0, 101):
        for yi in range(0, 101):
            x = (xi / 101) * 6
            y = (yi / 101) * 6
            pt = np.asarray([[x, y]])
            dist_r = distance.cdist(pt, red_pts, 'euclidean')
            dist_b = distance.cdist(pt, blue_pts, 'euclidean')
            r_cnt, b_cnt = 0, 0
            for k in range(3):
                min_r = np.argmin(dist_r)
                min_b = np.argmin(dist_b)
                if dist_r[0][min_r] < dist_b[0][min_b]:
                    dist_r = np.delete(dist_r, min_r, axis=1)
                    r_cnt += 1
                else:
                    dist_b = np.delete(dist_b, min_b, axis=1)
                    b_cnt += 1
            if r_cnt > b_cnt:
                plt.scatter(x, y, color='m', marker='s', s=3)
            else:
                plt.scatter(x, y, color='c', marker='s', s=3)

    plt.scatter(red_pts[:, 0], red_pts[:, 1], color='r')
    plt.scatter(blue_pts[:, 0], blue_pts[:, 1], color='b')

    plt.show()


if __name__ == '__main__':
    my_knn()
