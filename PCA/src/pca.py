#!/usr/bin/env python
import numpy as np

__author__ = "Wei-Chung Huang"
__copyright__ = "Copyright 2017"
__version__ = "1.0.0"


def project_onto_eig_space(A, eig_pair, num_eig):
    """
    Project MxN matrix A onto eigen-space
    :param A: MxN matrix where M is feature length and N is sample size
    :param eig_pair: A sorted list of tuple of (eigenvalue, eigenvector with size Mx1)
    :param num_eig: Number of the most dominant eigenvector
    :return: A projection space in size of num_eig x N
    """
    # Compute scoring matrix
    delta_mat = np.zeros(shape=(num_eig, A.shape[1]), dtype=np.float32)
    for i in range(num_eig):
        e_vec = eig_pair[i][1]
        for j in range(A.shape[1]):
            delta_mat[i, j] = A[:, j].T.dot(e_vec)

    return delta_mat


def train(training_set, num_eig):
    """
    Train a PCA model to get a scoring matrix
    :param training_set: MxN matrix where M is feature length and N is sample size
    :param num_eig: Number of the most dominant eigenvector
    :return: (scoring matrix in num_eig x N, list of eigen-pairs(eigval, eigvec in column vector), training means)
    """
    training_size = training_set.shape[1]
    # Zero mean
    A = np.asmatrix(np.zeros(shape=training_set.shape))
    training_means = [0] * A.shape[0]
    for i in range(A.shape[0]):
        training_means[i] = np.mean(training_set[i]);
        A[i] = training_set[i] - training_means[i]

    # Covariance matrix
    cov_mat = A * A.T / training_size

    # Compute the eigenvalue and eigenvector
    eig_val, eig_vec = np.linalg.eig(cov_mat)
    eig_val = eig_val.real
    eig_vec = eig_vec.real

    # Sort the eigen-pairs with key = eigenvalue
    eig_pair = [(eig_val[i], eig_vec[:, i]) for i in range(len(eig_val))]
    eig_pair.sort(key=lambda x: x[0], reverse=True)

    # Compute scoring matrix
    A_eig = project_onto_eig_space(A, eig_pair, num_eig)

    return A_eig, eig_pair, training_means


def score(testing_set, scoring_mat, eig_pair, num_eig, training_means):
    """
    Score the testing samples.
    :param testing_set: MxN matrix where M is feature length and N is sample size
    :param scoring_mat: Scoring matrix(num_eig x N) from the training phase
    :param eig_pair: Eigen-pairs from the training phase
    :param num_eig: Number of the most dominant eigenvector (has to fit with the training phase)
    :param training_means: Training means from the training phase
    :return: List of minimum scores(distances) for each testing sample
    """
    training_size = scoring_mat.shape[1]
    for i in range(Y.shape[0]):
        Y[i] = testing_set[i] - training_means[i]

    Y_eig = project_onto_eig_space(Y, eig_pair, num_eig)

    # For each testing vector
    scores = [0] * Y.shape[1]
    for i in range(Y.shape[1]):
        # Compute the distance of each one
        min_dist = np.finfo(np.float64).max
        for j in range(training_size):
            min_dist = min(min_dist, np.linalg.norm(scoring_mat[:, j] - Y_eig[:, i]))

        scores[i] = min_dist

    return scores


if __name__ == "__main__":

    # TEST 1 ##########################################
    x1 = np.matrix([2, -1, 0, 1, 1, -3, 5, 2]).T
    x2 = np.matrix([-2, 3, 2, 3, 0, 2, -1, 1]).T
    x3 = np.matrix([-1, 3, 3, 1, -1, 4, 5, 2]).T
    x4 = np.matrix([3, -1, 0, 3, 2, -1, 3, 0]).T
    eig_rank = 3

    A = np.matrix(np.stack((np.asarray(x1), np.asarray(x2), np.asarray(x3), np.asarray(x4)), axis=-1), dtype=np.float32)

    scoring_mat, eig_pair, training_means = train(A, eig_rank)

    y1 = np.matrix([1, 5, 1, 5, 5, 1, 1, 3]).T
    y2 = np.matrix([-2, 3, 2, 3, 0, 2, -1, 1]).T
    y3 = np.matrix([2, -3, 2, 3, 0, 0, 2, -1]).T
    y4 = np.matrix([2, -2, 2, 2, -1, 1, 2, 2]).T

    Y = np.matrix(np.stack((np.asarray(y1), np.asarray(y2), np.asarray(y3), np.asarray(y4)), axis=-1), dtype=np.float32)

    scores = score(Y, scoring_mat, eig_pair, eig_rank, training_means)

    # TEST 2 ##########################################
    u1 = np.matrix([[0.1641, 0.6278, -0.2604, -0.5389, 0.4637, 0.0752]]).T
    u2 = np.matrix([[0.2443, 0.1070, -0.8017, 0.4277, -0.1373, -0.2904]]).T
    eig_pair = [(4.0833, u1), (1.2364, u2)]

    m1 = np.matrix([1, -1, 1, -1, -1, 1]).T
    m2 = np.matrix([-2, 2, 2, -1, -2, 2]).T
    m3 = np.matrix([1, 3, 0, 1, 3, 1]).T
    m4 = np.matrix([2, 3, 1, 1, -2, 0]).T
    M = np.matrix(np.stack((np.asarray(m1), np.asarray(m2), np.asarray(m3), np.asarray(m4)), axis=-1), dtype=np.float32)
    mean_vec = [0] * 6
    for i in range(6):
        mean_vec[i] = np.mean(M[i])
        M[i] = M[i] - mean_vec[i]

    score_malware = project_onto_eig_space(M, eig_pair, 2)

    b1 = np.matrix([-1, 2, 1, 2, -1, 0]).T
    b2 = np.matrix([-2, 1, 2, 3, 2, 1]).T
    b3 = np.matrix([-1, 3, 0, 1, 3, -1]).T
    b4 = np.matrix([0, 2, 3, 1, 1, -2]).T
    B = np.matrix(np.stack((np.asarray(b1), np.asarray(b2), np.asarray(b3), np.asarray(b4)), axis=-1), dtype=np.float32)
    for i in range(6):
        B[i] = B[i] - mean_vec[i]

    score_benign = project_onto_eig_space(B, eig_pair, 2)

    y1 = np.matrix([1, 5, 1, 5, 5, 1]).T
    y2 = np.matrix([-2, 3, 2, 3, 0, 2]).T
    y3 = np.matrix([2, -3, 2, 3, 0, 0]).T
    y4 = np.matrix([2, -2, 2, 2, -1, 1]).T
    Y = np.matrix(np.stack((np.asarray(y1), np.asarray(y2), np.asarray(y3), np.asarray(y4)), axis=-1), dtype=np.float32)
    for i in range(6):
        Y[i] = Y[i] - mean_vec[i]

    score_testing = project_onto_eig_space(Y, eig_pair, 2)
    print(score_testing, '\n')

    for i in range(4):
        src = score_testing[:, i]
        dist_malware = np.finfo(np.float64).max
        for j in range(4):
            dst = score_malware[:, j]
            dist_malware = min(dist_malware, np.linalg.norm(src - dst))

        dist_benign = np.finfo(np.float64).max
        for j in range(4):
            dst = score_benign[:, j]
            dist_benign = min(dist_benign, np.linalg.norm(src - dst))

        if dist_malware < dist_benign:
            print("Malware")
        else:
            print("Benign")