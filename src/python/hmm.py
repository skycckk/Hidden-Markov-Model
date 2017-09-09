import re
import random
import math

if __name__ == "__main__":
    print("HMM testing itself.")


def train(file=None, *args):
    """
    Hidden Markov Model training entrance.
    :param file: file path of the corpus
    :param args: # of hidden states(N), # of observed symbols(M), # of observations(M), PRNG seed, max iteration
    :return:
    """
    if file is None or len(args) < 5:
        print("Parameters: file_path, N, M, T, seed, maxIter")
        return

    n_len = args[0]
    m_len = args[1]
    t_len = args[2]
    seed = args[3]
    max_iteration = args[4]

    corpus = [0] * t_len
    read_letters(file, t_len, corpus, True)

    a_matrix = [([0] * n_len) for i in range(n_len)]
    b_matrix = [([0] * m_len) for i in range(n_len)]
    pi_matrix = [0] * n_len
    model = {'A': a_matrix, 'B': b_matrix, 'pi': pi_matrix}
    observation = [0] * t_len

    init_model(n_len, m_len, model, seed)
    init_observation(observation, corpus)
    hmm_train(n_len, m_len, t_len, model, observation, max_iteration)


def init_observation(observation, corpus):
    """
    Initialize the observation in sense of english alphabet.
    Only lower case letter and space are treated as 27
    :param observation: observation list with length T
    :param corpus: list of all characters
    :return:
    """
    for i in range(len(observation)):
        if corpus[i] == ' ':
            observation[i] = 26
        else:
            observation[i] = ord(corpus[i]) - ord('a')


def read_letters(file, t_len, corpus, read_space=False):
    """
    Read alphabet from file to corpus
    :param file: file path
    :param t_len: length of corpus or length of observation
    :param corpus: output corpus list of chars
    :param read_space: a flag to read space or not
    :return:
    """
    count = 0
    with open(file) as f:
        for line in f:
            words = " ".join(re.findall("[a-zA-Z]+", line))
            for ch in words:
                if read_space is True and ch != ' ':
                    ch = ch.lower()
                corpus[count] = ch
                count = count + 1
                if count >= t_len:
                    break
            if count >= t_len:
                break


def init_model(n_len, m_len, model, seed):
    """
    Initialize lambda model with nearly an uniform distribution.
    Model = [A, B, pi]
    A: N x N
    B: N x M
    pi: 1 x N
    :param n_len: length of hidden states
    :param m_len: length of different observation symbols
    :param model: a dictionary with {'A': a_matrix, 'B': b_matrix, 'pi': pi_matrix}
    :param seed: initial random seed
    :return:
    """
    a = model['A']
    b = model['B']
    pi = model['pi']

    random.seed(seed)

    ceil_a = 1 / n_len
    ceil_b = 1 / m_len
    precision = 10

    for i in range(n_len):
        prob_sum = 0
        for j in range(n_len - 1):
            r = random.uniform(0, 1)
            if j % 2 == 0:
                a[i][j] = ceil_a + r * ceil_a / precision
            else:
                a[i][j] = ceil_a - r * ceil_a / precision
            prob_sum += a[i][j]
        a[i][n_len - 1] = 1 - prob_sum

        prob_sum = 0
        for j in range(m_len - 1):
            r = random.uniform(0, 1)
            if j % 2 == 0:
                b[i][j] = ceil_b + r * ceil_b / precision
            else:
                b[i][j] = ceil_b - r * ceil_b / precision
            prob_sum += b[i][j]
        b[i][m_len - 1] = 1 - prob_sum

    prob_sum = 0
    for i in range(n_len - 1):
        r = random.uniform(0, 1)
        if i % 2 == 0:
            pi[i] = ceil_a + r * ceil_a / precision
        else:
            pi[i] = ceil_a - r * ceil_a / precision
        prob_sum += pi[i]
    pi[n_len - 1] = 1 - prob_sum


def hmm_train(n_len, m_len, t_len, model, observation, max_iteration):
    """
    Kernel training entrance.
    Step1: alpha_pass
    Step2: beta_pass
    Step3: gamma_pass
    Step4: re_estimate
    Step5: evaluate model
    Step6: Goto step1 if needed
    :param n_len: length of hidden states
    :param m_len: length of different observed symbols
    :param t_len: length of observation sequence
    :param model: lambda model
    :param observation: list of observation sequence
    :param max_iteration: max iteration of HMM
    :return:
    """
    for i in range(n_len):
        test_sum = 0
        for j in range(m_len):
            test_sum += model['B'][i][j]
        print("sum = ", test_sum)

    scaling_factor = [0] * t_len
    alpha = [([0] * t_len) for i in range(n_len)]
    beta = [([0] * t_len) for i in range(n_len)]
    gamma = [([0] * t_len) for i in range(n_len)]
    di_gamma = [([([0] * t_len) for i in range(n_len)]) for i in range(n_len)]

    dump_model(n_len, m_len, model)

    old_prob = 0
    for i in range(max_iteration):
        alpha_pass(n_len, t_len, observation, alpha, model, scaling_factor)
        beta_pass(n_len, t_len, observation, beta, model, scaling_factor)
        gamma_pass(n_len, t_len, observation, alpha, beta, gamma, di_gamma, model)
        re_estimate(n_len, m_len, t_len, observation, gamma, di_gamma, model)
        log_prob = get_probability(t_len, scaling_factor)
        if i > 1 and log_prob < old_prob:
            break
        old_prob = log_prob
        print("log_prob: ", log_prob)
    dump_model(n_len, m_len, model)


def alpha_pass(n_len, t_len, observation, alpha, model, c):
    """
    HMM training step1: Forward Algorithm(alpha pass).
    :param n_len: length of hidden states
    :param t_len: length of observation sequence
    :param observation: list of observation
    :param alpha: output alpha list with size N x T
    :param model: HMM model
    :param c: scaling factor with size 1 x T
    :return:
    """
    a = model['A']
    b = model['B']
    pi = model['pi']
    c[0] = 0
    for i in range(0, n_len):
        alpha[i][0] = pi[i] * b[i][observation[0]]
        c[0] += alpha[i][0]

    c[0] = 1 / c[0]
    for i in range(0, n_len):
        alpha[i][0] *= c[0]

    for t in range(1, t_len):
        c[t] = 0
        for i in range(0, n_len):
            alpha[i][t] = 0
            for j in range(0, n_len):
                alpha[i][t] += alpha[j][t - 1] * a[j][i]
            alpha[i][t] *= b[i][observation[t]]
            c[t] += alpha[i][t]

        c[t] = 1.0 / c[t]
        for i in range(0, n_len):
            alpha[i][t] *= c[t]


def beta_pass(n_len, t_len, observation, beta, model, c):
    """
    HMM training step2: Backward Algorithm(beta pass)
    :param n_len: length of hidden states
    :param t_len: length of observation sequence
    :param observation: list of observation sequence
    :param beta: output beta list of size N x M
    :param model: HMM model
    :param c: scaling factor with size 1 x T
    :return:
    """
    a = model['A']
    b = model['B']
    for i in range(0, n_len):
        beta[i][t_len - 1] = c[t_len - 1]

    for t in range(t_len - 2, -1, -1):
        for i in range(0, n_len):
            beta[i][t] = 0
            for j in range(0, n_len):
                beta[i][t] += a[i][j] * b[j][observation[t + 1]] * beta[j][t + 1]
            beta[i][t] *= c[t]


def gamma_pass(n_len, t_len, observation, alpha, beta, gamma, di_gamma, model):
    """
    HMM training step3: Compute gamma and di_gamma.
    :param n_len: length of hidden states
    :param t_len: length of observation sequence
    :param observation: list of observation sequence
    :param alpha: the alpha vector from alpha bass
    :param beta: the beta vector from beta pass
    :param gamma: output gamma list with size N x T
    :param di_gamma: output di_gamma list with size N x N x T
    :param model: HMM model
    :return:
    """
    a = model['A']
    b = model['B']
    for t in range(t_len - 1):
        denom = 0
        for i in range(n_len):
            denom += alpha[i][t_len - 1];

        for i in range(n_len):
            gamma[i][t] = 0
            for j in range(n_len):
                di_gamma[i][j][t] = alpha[i][t] * a[i][j] * b[j][observation[t + 1]] * beta[j][t + 1] / denom
                gamma[i][t] += di_gamma[i][j][t]

    denom = 0
    for i in range(n_len):
        denom += alpha[i][t_len - 1]
    for i in range(n_len):
        gamma[i][t_len - 1] = alpha[i][t_len - 1] / denom


def re_estimate(n_len, m_len, t_len, observation, gamma, di_gamma, model):
    """
    HMM training step4: Re-estimate model A, B, and pi matrix.
    :param n_len: length of hidden states
    :param m_len: length of different observed symbols
    :param t_len: length of observation sequence
    :param observation: list of observation sequence
    :param gamma: the gamma vector from gamma_pass
    :param di_gamma: the di_gamma vector from gamma_pass
    :param model: HMM model
    :return:
    """
    a = model['A']
    b = model['B']
    pi = model['pi']

    for i in range(n_len):
        pi[i] = gamma[i][0]

    for i in range(n_len):
        for j in range(n_len):
            numer = 0
            denom = 0
            for t in range(t_len - 1):
                numer += di_gamma[i][j][t]
                denom += gamma[i][t]
            a[i][j] = numer / denom

        for k in range(m_len):
            numer = 0
            denom = 0
            for t in range(t_len):
                if observation[t] == k:
                    numer += gamma[i][t]
                denom += gamma[i][t]
            b[i][k] = numer / denom


def get_probability(t_len, scaling):
    """
    HMM training step5: Evaluate new observation probability given an HMM model
    :param t_len: length of observation sequence
    :param scaling: scaling factor
    :return: logarithm probability
    """
    log_prob = 0
    for i in range(t_len):
        log_prob += math.log(scaling[i])
    log_prob = -log_prob
    return log_prob


def dump_model(n_len, m_len, model):
    """
    Print the model values for debug.
    :param n_len: length of hidden states
    :param m_len: length of different observations
    :param model: HMM model
    :return:
    """
    print("\n----A matrix----")
    for i in range(n_len):
        for j in range(n_len): print("%.4f" % model['A'][i][j], " ", end="")
        print()
    print("\n----B matrix----")
    for i in range(n_len):
        for j in range(m_len): print("%.4f" % model['B'][i][j], " ", end="")
        print()
    print("\n----pi matrix----")
    for i in range(n_len): print("%.4f" % model['pi'][i], " ", end="")
    print()