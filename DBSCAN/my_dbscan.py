import numpy as np
import matplotlib.pyplot as plt

__author__ = "Wei-Chung Huang"
__copyright__ = "Copyright 2017"
__version__ = "1.0.0"

def my_dbscan(db, eps, min_pts):
    """
    A simple implementation of DBSCAN
    :param db: ndarray
            Data points in row vectors.
    :param eps: float
            An epsilon. This determines the reachable points.
    :param min_pts: int
            Density checking. A threshold of deciding core points.
    :return: list
            Cluster result where in index is an index of input data and value is cluster number.
    """
    label = [0] * len(db)
    cluster = 0
    for i in range(len(db)):
        p = db[i]
        if label[i] is not 0:
            continue

        neighbors = range_query(db, eps, p)
        if len(neighbors) < min_pts:
            label[i] = -1
            continue

        cluster += 1
        label[i] = cluster
        j = 0
        while j < len(neighbors):
            q = neighbors[j]
            if label[q] is -1:
                label[q] = cluster
            elif label[q] is 0:
                label[q] = cluster
                neighbors_tmp = range_query(db, eps, q)
                if len(neighbors_tmp) >= min_pts:
                    neighbors = neighbors + neighbors_tmp

            j += 1
    return label


def range_query(db, eps, p):
    """
    List all neighbor points.
    :param db: ndarray
            Data points in row vectors.
    :param eps: float
            An epsilon. This determines the reachable points.
    :param p: ndarray
            Referred point to find its neighbors.
    :return: list
            List of neighbors in index form of data.
    """
    neighbors = list()
    for i in range(len(db)):
        q = db[i]
        if np.linalg.norm(p - q) < eps:
            neighbors.append(i)

    return neighbors


def test():
    data = [[1.0, 5.0], [1.25, 5.35], [1.25, 5.75], [1.5, 6.25], [1.75, 6.75],
            [2.0, 6.5], [3.0, 7.75], [3.5, 8.25], [3.75, 8.75], [3.95, 9.1],
            [4.0, 8.5], [2.5, 7.25], [2.25, 7.75], [2.0, 6.5], [2.75, 8.25],
            [4.5, 8.9], [9.0, 5.0], [8.75, 5.85], [9.0, 6.25], [8.0, 7.0],
            [8.5, 6.25], [8.5, 6.75], [8.25, 7.65], [7.0, 8.25], [6.0, 8.75],
            [5.5, 8.25], [5.25, 8.75], [4.9, 8.75], [5.0, 8.5], [7.5, 7.75],
            [7.75, 8.25], [6.75, 8.0], [6.25, 8.25], [4.5, 8.9], [5.0, 1.0],
            [1.25, 4.65], [1.25, 4.25], [1.5, 3.75], [1.75, 3.25], [2.0, 3.5],
            [3.0, 2.25], [3.5, 1.75], [3.75, 8.75], [3.95, 0.9], [4.0, 1.5],
            [2.5, 2.75], [2.25, 2.25], [2.0, 3.5], [2.75, 1.75], [4.5, 1.1],
            [5.0, 9.0], [8.75, 5.15], [8.0, 2.25], [8.25, 3.0], [8.5, 4.75],
            [8.5, 4.25], [8.25, 3.35], [7.0, 1.75], [8.0, 3.5], [6.0, 1.25],
            [5.5, 1.75], [5.25, 1.25], [4.9, 1.25], [5.0, 1.5], [7.5, 2.25],
            [7.75, 2.75], [6.75, 2.0], [6.25, 1.75], [4.5, 1.1], [3.0, 4.5],
            [7.0, 4.5], [5.0, 3.0], [4.0, 3.35], [6.0, 3.35], [4.25, 3.25],
            [5.75, 3.25], [3.5, 3.75], [6.5, 3.75], [3.25, 4.0], [6.75, 4.0],
            [3.75, 3.55], [6.25, 3.55], [4.75, 3.05], [5.25, 3.05], [4.5, 3.15],
            [5.5, 3.15], [4.0, 6.5], [4.0, 6.75], [4.0, 6.25], [3.75, 6.5],
            [4.25, 6.5], [4.25, 6.75], [3.75, 6.25], [6.0, 6.5], [6.0, 6.75],
            [6.0, 6.25], [5.75, 6.75], [5.75, 6.25], [6.25, 6.75], [6.25, 6.25],
            [9.5, 9.5], [2.5, 9.5], [1.0, 8.0]]

    data = np.asarray(data)
    labels = my_dbscan(data, 0.6, 3)
    # labels = my_dbscan(data, 0.75, 4)
    # labels = my_dbscan(data, 1.0, 5)
    # labels = my_dbscan(data, 2.0, 10)
    clusters = set(labels)
    n_clusters = len(clusters)
    # for i in clusters:
    #     print(i, labels.count(i))

    color_labels = [0] * n_clusters
    color_ch_label = [0] * n_clusters
    np.random.seed(1)
    for i in range(n_clusters):
        scale = int(255 / n_clusters)
        color_ch_label[i] = np.random.randint(0, 3)
        color_labels[i] = (255 - (scale * i)) / 255
    for i in range(data.shape[0]):
        cl = [.0, .0, .0]
        cluster_idx = labels[i]
        if cluster_idx is -1:
            plt.scatter(data[i, 0], data[i, 1], color='black', marker='*')
        else:
            color = color_labels[cluster_idx]
            cl[color_ch_label[cluster_idx]] = color
            cl = tuple(cl)
            plt.scatter(data[i, 0], data[i, 1], color=cl, s=10)
    plt.title('Total clusters ' + str(n_clusters) + '(include outliers)')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

if __name__ == '__main__':
    test()
