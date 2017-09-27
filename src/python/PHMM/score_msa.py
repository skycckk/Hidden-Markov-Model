### Author: Wei-Chung Huang
### CS286
### Test for computing MSA scores in PHMM

def compute_msa_score(X, Y, s, letter_index_dict):
    """
    comptute the MSA score by using Dynamic Programming
    :param X: Input sequence 1
    :param Y: Input sequence 2
    :param s: substitution matrix
    :param letter_index_dict: a dictionary {sequence symbol: s_matrix index}
    :return: F(i, j) matrix that indicates the optimal score aligned from 1 to i and 1 to j
    """
    n, m = len(X), len(Y)
    F = [([0] * (m + 1)) for i in range(n + 1)] # (n + 1) x (m + 1)
    G = [([0] * (m + 1)) for i in range(n + 1)] # (n + 1) x (m + 1)

    for i in range(n + 1): G[i][0], F[i][0] = 0, 0
    for j in range(m + 1): 
        G[0][j], sum = 0, 0
        for k in range(j + 1): sum += penalty_cost(k)
        F[0][j] = sum

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            x_val, y_val = letter_index_dict[X[i - 1]], letter_index_dict[Y[j - 1]]
            case1 = F[i - 1][j - 1] + s[x_val][y_val]
            case2 = F[i - 1][j] + penalty_cost(G[i - 1][j])
            case3 = F[i][j - 1] + penalty_cost(G[i][j - 1])
            if case1 > case2 and case1 > case3:
                G[i][j], F[i][j] = 0, case1
            elif case2 > case1 and case2 > case3:
                G[i][j], F[i][j] = G[i - 1][j] + 1, case2
            elif case3 > case2 and case3 > case1:
                G[i][j], F[i][j] = G[i][j - 1] + 1, case3

    return F


def penalty_cost(x):
    return -3 * x


if __name__ == "__main__":
    X = ['E', 'J', 'G']
    Y = ['G', 'E', 'E'] #, 'C', 'G']
    training = [['E', 'J', 'G'], 
                ['G', 'E', 'E', 'C', 'G'],
                ['C', 'G', 'J', 'E', 'E'],
                ['J', 'J', 'G', 'E', 'C', 'C', 'G']]
    substituion_score = [[ 9, -4,  2,  2],
                         [-4,  9, -5, -5],
                         [ 2, -5, 10,  7],
                         [ 2, -5,  7, 10]]
    letter_index_dict = {'E': 0, 'G': 1, 'C': 2, 'J': 3}

    pair_align_scores = [([0] * len(training)) for i in range(len(training))]
    for i in range(len(training) - 1):
        for j in range(i + 1, len(training)):
            X, Y = training[i], training[j]
            F = compute_msa_score(X, Y, substituion_score, letter_index_dict)
            max_score = 0
            for p in range(len(X)):
                for q in range(len(Y)):
                    max_score = max(F[p + 1][q + 1], max_score)
            pair_align_scores[i][j] = max_score
            pair_align_scores[j][i] = max_score

    ### Print the pairwaise alignment scores
    for i in range(len(pair_align_scores)):
        for j in range(len(pair_align_scores)):
            print("%2d" % pair_align_scores[i][j], end=' ')
        print()
        
