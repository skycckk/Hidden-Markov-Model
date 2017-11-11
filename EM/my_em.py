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


def e_step(data, theta, tao, n_clusters, n_samples):
    p = np.zeros([n_clusters, n_samples])
    for i in range(n_samples):
        bayes_sum = 0
        x = data[i]
        for j in range(n_clusters):
            bayes_sum += tao[j] * prob_func(x, theta[j], 10)

        for j in range(n_clusters):
            p[j][i] = tao[j] * prob_func(x, theta[j], 10) / bayes_sum

    return p


def m_step(data, p, tao, n_clusters, n_samples):
    pj_sum = p.sum(axis=1)
    mu = np.zeros(n_clusters)
    var = np.zeros(n_clusters)
    # re-estimate mean and variance
    for j in range(n_clusters):
        tao[j] = pj_sum[j] / n_samples
        mu[j] = (p[j] * data).sum() / pj_sum[j]
        var[j] = (p[j] * (data - mu[j]) ** 2).sum() / pj_sum[j]

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
        p = e_step(x, theta, tao, n_clusters, n_samples)
        mu = m_step(x, p, tao, n_clusters, n_samples)
        old_theta = theta.copy()
        for j in range(n_clusters):
            theta[j] = mu[j] / 10

        iter += 1
        print('Iteration:', iter, '/', max_iter)
        print('old theta', old_theta)
        print('new theta', theta, '\n')
        dist = np.linalg.norm(theta - old_theta)

if __name__ == '__main__':
    test()
