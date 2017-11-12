import numpy as np
from math import factorial

__author__ = "Wei-Chung Huang"
__copyright__ = "Copyright 2017"
__version__ = "1.0.0"


def prob_func(x, theta, n):
    """
    Binomial Distribution
    :param x: int
            The outcome value
    :param theta: float
            The probability at desired Head or Tail
    :param n: int
            Number of trials
    :return: float
            The probability of binomial
    """
    k = x
    a = factorial(n)
    b = factorial(k)
    c = factorial(n - k)
    term1 = a / b * c
    term2 = theta ** k
    term3 = (1.0 - theta) ** (n - k)
    return term1 * term2 * term3


def bi_gaussian(x, theta):
    """
    Bivariate Gaussian Distribution
    :param x: ndarray
            m-by-1 array
    :param theta: dictionary
            key['mu']: m-by-1 array
            key['S']: m-by-m covariance matrix S
    :return:
    """
    mu = theta['mu']
    s = theta['S']
    term1 = 2.0 * np.pi * np.sqrt(np.linalg.det(s))
    term2 = -((x - mu).T.dot(np.linalg.inv(s)).dot(x - mu)) / 2.0
    term2 = np.exp(term2)
    return term2 / term1


def e_step_1d(data, theta, tao, n_clusters, n_samples):
    p = np.zeros([n_clusters, n_samples])
    for i in range(n_samples):
        bayes_sum = 0
        x = data[i]
        for j in range(n_clusters):
            bayes_sum += tao[j] * prob_func(x, theta[j], 10)

        for j in range(n_clusters):
            p[j][i] = tao[j] * prob_func(x, theta[j], 10) / bayes_sum

    return p


def m_step_1d(data, p, tao, n_clusters, n_samples):
    pj_sum = p.sum(axis=1)
    mu = np.zeros(n_clusters)
    var = np.zeros(n_clusters)
    # re-estimate mean and variance
    for j in range(n_clusters):
        tao[j] = pj_sum[j] / n_samples
        mu[j] = (p[j] * data).sum() / pj_sum[j]
        var[j] = (p[j] * (data - mu[j]) ** 2).sum() / pj_sum[j]

    return mu


def e_step(data, theta, tao, n_clusters, n_samples):
    """
    E-step. (Expectation Steps)
    :param data: ndarray
            Column vector.
            m-by-n array where m is dimension and n is no. samples.
    :param theta: dictionary
            key['mu']: m-by-1 array
            key['S']: m-by-m covariance matrix S
    :param tao: ndarray
            The probability for each clusters.
            1-by-c array
    :param n_clusters: int
            Number of clusters.
    :param n_samples: int
            Number of sample size.
    :return: ndarray
            c-by-n array where c is the no. clusters.
    """
    p = np.zeros([n_clusters, n_samples])  # j, i index
    for i in range(n_samples):
        bayes_sum = 0
        x = data[:, [i]]
        for j in range(n_clusters):
            bayes_sum += tao[j] * bi_gaussian(x, theta[j])

        for j in range(n_clusters):
            p[j][i] = tao[j] * bi_gaussian(x, theta[j]) / bayes_sum

    return p


def m_step(data, p, tao, n_clusters, n_samples):
    """
    M-step. (Maximization Steps)
    :param data: ndarray
            Column vector.
            m-by-n array where m is dimension and n is no. samples.
    :param p: ndarray
            Probability from te E-step.
            c-by-n array where c is the no. clusters.
    :param tao: ndarray
            The probability for each clusters.
            1-by-c array
    :param n_clusters: int
            Number of clusters.
    :param n_samples: int
            Number of samples
    :return: ndarray
            New mean column vectors with m-by-c where m is dimensionality.
    """
    pj_sum = p.sum(axis=1)
    n_dim = 2
    mu = np.zeros([n_dim, n_clusters])
    for j in range(n_clusters):
        tao[j] = pj_sum[j] / n_samples
        mu[:, j] = p[j].dot(data.T) / pj_sum[j]

    return mu


def test():
    n_clusters = 2
    n_samples = 5
    x = np.asarray([8, 5, 9, 4, 7])
    theta = np.asarray([0.6, 0.5])
    tao = np.asarray([0.7, 0.3])

    iter = 1
    max_iter = 100
    stopping_threshold = 10e-5
    dist = stopping_threshold + 1
    while iter <= max_iter and dist > stopping_threshold:
        p = e_step_1d(x, theta, tao, n_clusters, n_samples)
        mu = m_step_1d(x, p, tao, n_clusters, n_samples)
        old_theta = theta.copy()
        for j in range(n_clusters):
            theta[j] = mu[j] / 10

        iter += 1
        print('Iteration:', iter, '/', max_iter)
        print('old theta', old_theta)
        print('new theta', theta, '\n')
        dist = np.linalg.norm(theta - old_theta)


def test2():
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
    old_faithful_data = np.asarray(old_faithful_data).transpose()

    n_dim = 2
    n_clusters = 2
    n_samples = 20
    x = old_faithful_data
    mu = [np.asarray([2.5, 65]).reshape(n_clusters, -1),
          np.asarray([3.5, 70]).reshape(n_clusters, -1)]
    s = [np.asarray([[1, 5], [5, 100]]),
         np.asarray([[2, 10], [10, 200]])]
    theta = [{'mu': mu[0], 'S': s[0]},
             {'mu': mu[1], 'S': s[1]}]
    tao = np.asarray([0.6, 0.4])

    iter = 1
    max_iter = 2
    while iter <= max_iter:
        p = e_step(x, theta, tao, n_clusters, n_samples)
        new_mu = m_step(x, p, tao, n_clusters, n_samples)

        # re-estimate S
        pj_sum = p.sum(axis=1)
        for j in range(n_clusters):
            x_mu = x - new_mu[:, [j]]
            new_s = np.zeros([n_dim, n_dim])
            for i in range(n_samples):
                new_s += p[j][i] * (x_mu[:, [i]]).dot((x_mu[:, [i]]).T)
            new_s = new_s / pj_sum[j]
            s[j] = new_s

        theta = [{'mu': new_mu[:, [0]], 'S': s[0]},
                 {'mu': new_mu[:, [1]], 'S': s[1]}]

        print('Iteration:', iter, '/', max_iter)
        print('tau:\n', tao)
        print('mu:\n', new_mu)
        print()
        iter += 1

if __name__ == '__main__':
    test()
    test2()
