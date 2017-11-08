import numpy as np
import matplotlib.pyplot as plt

__author__ = "Wei-Chung Huang"
__copyright__ = "Copyright 2017"
__version__ = "1.0.0"


def _distance(a, b):
    """
    Distance function. Use Euclidean distance so far.
    :param a: ndarray
        Set of data points
    :param b: ndarray
        Set of data points
    :return: float
        Distance of a and b
    """
    return np.linalg.norm(a - b)


def labeling(data, centroids, labels):
    """
    Label each sample with its nearest centroids.
    The label starts from 0 to k - 1
    :param data: ndarray
            N-by-M ndarray where N is the number of samples
    :param centroids: ndarray
            K-by-M ndarray where K is the number of centroids
    :param labels: ndarray
            1D ndarray where length is same as number of samples
    :return:
    """
    for i in range(data.shape[0]):
        sample = data[i, :]
        # every sample
        min_dist = np.finfo(np.float32).max
        label = 0
        for k in range(centroids.shape[0]):
            centroid = centroids[k, :]
            dist = _distance(sample, centroid)
            if dist < min_dist:
                min_dist = dist
                label = k
        labels[i] = label


def my_kmeans(data, k):
    """
    Simple k-means algorithm
    :param data: ndarray
            Input data.
            N-by-M ndarray where N is the number of samples
    :param k: int
            number of clusters
    :return: ndarray
            Output centroids
            K-by-M ndarray where K is the number of centroids
    """
    n_samples = data.shape[0]
    labels = np.zeros(data.shape[0], dtype=np.int8)

    # initialize by random
    np.random.seed(1)
    rand_sample_idx = np.random.choice(n_samples, k, replace=False)
    centroids = data[rand_sample_idx]
    old_centroids = centroids.copy()
    labeling(data, centroids, labels)

    iteration = 0
    threshold = 0.00002
    stopping_dist = threshold + 1
    while iteration < 1000 and stopping_dist > threshold:
        # re-compute centroids
        clusters = [list() for i in range(k)]
        for i in range(n_samples):
            cluster = clusters[labels[i]]
            cluster.append(list(data[i]))

        # assign new centroids
        for i in range(k):
            cluster = np.asarray(clusters[i])
            centroids[i] = np.mean(cluster, axis=0, dtype=np.float32)

        # labeling
        labeling(data, centroids, labels)

        iteration += 1

        # compute stopping criteria
        stopping_dist = _distance(centroids, old_centroids)
        old_centroids = centroids.copy()

    print('labels\n', labels)
    print('centroids\n', centroids)
    return centroids


if __name__ == '__main__':
    old_faithful_data = [[3.600, 79],
                         [1.800, 54],
                         [2.283, 62],
                         [3.333, 74],
                         [2.883, 55],
                         [4.533, 85],
                         [1.950, 51],
                         [1.833, 54],
                         [4.700, 88],
                         [3.600, 85],
                         [1.600, 52],
                         [4.350, 85],
                         [3.917, 84],
                         [4.200, 78],
                         [1.750, 62],
                         [1.800, 51],
                         [4.700, 83],
                         [2.167, 52],
                         [4.800, 84],
                         [1.750, 47]]
    old_faithful_data = np.asarray(old_faithful_data)
    clusters = my_kmeans(old_faithful_data, 3)

    plt.scatter(old_faithful_data[:, 0], old_faithful_data[:, 1])
    plt.title('Old Faithful Data with K = 2')
    for centroid in clusters:
        plt.scatter(centroid[0], centroid[1], color='red')
    plt.show()

