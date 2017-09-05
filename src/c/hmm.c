#include "stdio.h"
#include "stdlib.h"
#include "time.h"
#include "math.h"
#include <ctype.h>

#include "mycipher.h"

void hmm_train(int M, int N, int T, int *O, int max_iter);
void init_model(int M, int N);
void dump_model(int M, int N);
double **alpha_pass(int N, int T, int *O);
double **alpha_pass_no_scale(int N, int T, int *O);
double **beta_pass(int N, int T, int *O);
double ***compute_gamma(int N, int T, int *O, double **alpha, double **beta, double ***gamma);
void re_estimate(int N, int M, int T, int *O, double **gamma, double ***di_gamma);
double compute_prob(int T);
void init_brown_corpus(int T);
void init_simple_corpus(int T);
double get_all_states_prob_sum(int N, int T, int *O, int i, int t, double prev_prob);
double get_all_observations_prob_sum(int M, int N, int T, int t, int *O, int use_alpha_pass);
int verify_all_prob_is_valid(int M, int N, int T, int *O);

double **A = NULL;
double **B = NULL;
double *pi = NULL;

int *O = NULL;
double *C = NULL;

void dump_model(int M, int N) {
    // dump A-matrix
    printf("\n--------A-Matrix--------\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", A[i][j]);
        }
        printf("\n");
    }

    printf("\n--------B-Matrix--------\n");
    for (int i = 0; i < M; i++) {
        if (i < M) printf("    %c | ", i + 'a');
        // else printf("space | ");
        for (int j = 0; j < N; j++) {
            printf("%f ", B[j][i]);
        }
        printf("\n");
    }

    printf("\n--------pi-Matrix--------\n");
    for (int i = 0; i < N; i++) {
        printf("%f ", pi[i]);
    }
    printf("\n");
}

void init_simple_corpus(int T) {
    if (T != 4) {fprintf(stderr, "ERROR: wrong T value."); exit(1);};

    A[0][0] = 0.7; A[0][1] = 0.3;
    A[1][0] = 0.4; A[1][1] = 0.6;

    B[0][0] = 0.1; B[0][1] = 0.4; B[0][2] = 0.5;
    B[1][0] = 0.7; B[1][1] = 0.2; B[1][2] = 0.1;

    pi[0] = 0.6; pi[1] = 0.4;

    O = (int *)malloc(sizeof(int) * T);
    O[0] = 0; O[1] = 1;
    O[2] = 0; O[3] = 2;
}

void init_brown_corpus(int T) {
    A[0][0] = 0.47468;
    A[0][1] = 0.52532;
    A[1][0] = 0.51656;
    A[1][1] = 0.48344;

    B[0][0] = 0.03735; B[1][0] = 0.03909;
    B[0][1] = 0.03408; B[1][1] = 0.03537;
    B[0][2] = 0.03455; B[1][2] = 0.03537;
    B[0][3] = 0.03828; B[1][3] = 0.03909;
    B[0][4] = 0.03782; B[1][4] = 0.03583;
    B[0][5] = 0.03922; B[1][5] = 0.03630;
    B[0][6] = 0.03688; B[1][6] = 0.04048;
    B[0][7] = 0.03408; B[1][7] = 0.03537;
    B[0][8] = 0.03875; B[1][8] = 0.03816;
    B[0][9] = 0.04062; B[1][9] = 0.03909;
    B[0][10] = 0.03735; B[1][10] = 0.03490;
    B[0][11] = 0.03968; B[1][11] = 0.03723;
    B[0][12] = 0.03548; B[1][12] = 0.03537;
    B[0][13] = 0.03735; B[1][13] = 0.03909;
    B[0][14] = 0.04062; B[1][14] = 0.03397;
    B[0][15] = 0.03595; B[1][15] = 0.03397;
    B[0][16] = 0.03641; B[1][16] = 0.03816;
    B[0][17] = 0.03408; B[1][17] = 0.03676;
    B[0][18] = 0.04062; B[1][18] = 0.04048;
    B[0][19] = 0.03548; B[1][19] = 0.03443;
    B[0][20] = 0.03922; B[1][20] = 0.03537;
    B[0][21] = 0.04062; B[1][21] = 0.03955;
    B[0][22] = 0.03455; B[1][22] = 0.03816;
    B[0][23] = 0.03595; B[1][23] = 0.03723;
    B[0][24] = 0.03408; B[1][24] = 0.03769;
    B[0][25] = 0.03408; B[1][25] = 0.03955;
    B[0][26] = 0.03685; B[1][26] = 0.03394;

    pi[0] = 0.51316;
    pi[1] = 0.48484;

    // init observations
    O = (int *)malloc(sizeof(int) * T);

    char * line = NULL;
    size_t len = 0;
    FILE *fp = fopen("./dataset/brown.txt", "r");
    if (fp == NULL) {fprintf(stderr, "ERROR: Couldn't open file.\n"); exit(1);};

    ssize_t read;
    int count = 0;
    while ((read = getline(&line, &len, fp)) != -1){
        for (int i = 0; i < len; i++) {
            char c = line[i];
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == ' ') {
                if (c != ' ') O[count] = tolower(c) - 'a';
                else O[count] = 26;
                count++;
            }
            if (count >= T) break;
        }
        if (count >= T) break;
    }
    if (count < T) exit(0);    
}

void init_ss_cipher_corpus(int T) {
    int cipher_length = 0;
    char *cipher = gen_cipher_with_shifting(&cipher_length);
    if (T != cipher_length) {fprintf(stderr, "ERROR: Wrong size of T %d.\n", cipher_length); exit(0);};
    O = (int *)malloc(sizeof(int) * T);
    for (int i = 0; i < cipher_length; i++) O[i] = cipher[i] - 'a';
    free(cipher);        
}

double get_all_states_prob_sum(int N, int T, int *O, int i, int t, double prev_prob) {
    if (t >= T) {
        return prev_prob;
    }

    double prob = 0.0;
    double curr_prob = 0.0;
    for (int j = 0; j < N; j++) {
        if (t == 0) curr_prob = pi[j] * B[j][O[t]];
        else curr_prob = prev_prob * (A[i][j] * B[j][O[t]]);

        prob += get_all_states_prob_sum(N, T, O, j, t + 1, curr_prob);

        curr_prob = prev_prob;
    }
    return prob;
}

double get_all_observations_prob_sum(int M, int N, int T, int t, int *O, int use_alpha_pass) {
    if (t >= T) {
        double sum = 0.0;
        if (use_alpha_pass) {
            double **alpha = alpha_pass_no_scale(N, T, O);
            for (int i = 0; i < N; i++) sum += alpha[i][T - 1];
            if (alpha) {
                for (int i = 0; i < N; i++) free(alpha[i]);
                free(alpha);
            }
        } else {
            sum = get_all_states_prob_sum(N, T, O, 0, 0, 1.0);
        }
        return sum;
    }

    double sum = 0.0;
    for (int i = 0; i < M; i++) {
        O[t] = i;
        sum += get_all_observations_prob_sum(M, N, T, t + 1, O, use_alpha_pass);
    }
    return sum;
}

int verify_all_prob_is_valid(int M, int N, int T, int *O) {
    // verify all probability are sum to one
    // method 1: using original method
    double sum1 = get_all_observations_prob_sum(M, N, T, 0, O, 0);
    // method 2: using alpha-pass
    double sum2 = get_all_observations_prob_sum(M, N, T, 0, O, 1);

    return (fabs(sum1 - 1.0) < 10e-7) && (fabs(sum2 - 1.0) < 10e-7);
}

int main(int argc, char *argv[]) {
    const int M = 26;
    const int N = 2;
    const int T = 50000;
    const int max_iter = 1000;
    init_model(M, N);
    
    init_ss_cipher_corpus(T);
    // init_simple_corpus(T);
    // init_brown_corpus(T);
    hmm_train(M, N, T, O, max_iter);

    if (A) {
        for (int i = 0; i < N; i++) free(A[i]);
        free(A);
    }
    if (B) {
        for (int i = 0; i < N; i++) free(B[i]);
        free(B);
    }
    if (pi) free(pi);
    if (O) free(O);
    return 0;
}

void hmm_train(int M, int N, int T, int* O, int max_iter) {
    printf("Start HMM Training...\n");
    // check input matrix
    for (int i = 0; i < N; i++) {
        double sum = 0.0;
        for (int j = 0; j < M; j++) sum += B[i][j];
        if (fabs(sum - 1.0) > 10e-7) { fprintf(stderr, "ERROR: B matrix is not stochastic (%f)\n", sum); exit(1); };

        sum = 0.0;
        for (int j = 0; j < N; j++) sum += A[i][j];
        if (fabs(sum - 1.0) > 10e-7) { fprintf(stderr, "ERROR: A matrix is not stochastic (%f)\n", sum); exit(1); };
    }

    dump_model(M, N);

    double old_prob = -(1.0 / 0.0); // to make it -INF
    double new_prob = 1.0;
    int iteration = 0;
    while (iteration < max_iter && new_prob > old_prob) {
        double **alpha = alpha_pass(N, T, O);
        double **beta = beta_pass(N, T, O);
        double **gamma = NULL;
        double ***di_gamma = compute_gamma(N, T, O, alpha, beta, &gamma);
        re_estimate(N, M, T, O, gamma, di_gamma);
        double new_prob = compute_prob(T);        
        old_prob = new_prob;
        iteration++;

        printf("----------------\n");
        printf("iter: %d, prob: %f\n", iteration, new_prob);
        // free memory
        if (alpha) {
            for (int i = 0; i < N; i++) free(alpha[i]);
            free(alpha);
        }

        if (beta) {
            for (int i = 0; i < N; i++) free(beta[i]);
            free(beta);
        }

        if (gamma) {
            for (int i = 0; i < N; i++) free(gamma[i]);
            free(gamma);   
        }

        if (di_gamma) {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    free(di_gamma[i][j]);
                }
                free(di_gamma[i]);
            }
            free(di_gamma);
        }

        free(C);
    }
    dump_model(M, N);
}

void init_model(int M, int N) {
    // A: NxN
    // B: NxM
    // pi: 1xN
    // Default is initializing to a nearly uniform distribution
    // buffer allocation
    A = (double **)malloc(sizeof(double*) * N);
    for (int i = 0; i < N; i++) A[i] = (double *)malloc(sizeof(double) * N);

    B = (double **)malloc(sizeof(double *) * N);
    for (int i = 0; i < N; i++) B[i] = (double *)malloc(sizeof(double) * M);

    pi = (double *)malloc(sizeof(double *) * N);

    // initialize value (nearly uniform distribution)
    srand(1);
    for (int i = 0; i < N; i++) {
        double sum = 0.0;
        for (int j = 0; j < N - 1; j++) {
            float r = (rand() % 100) / 100000.f;
            if (j % 2 == 0) A[i][j] = (1.0 / N) - r;
            else A[i][j] = (1.0 / N) + r;
            sum += A[i][j];
        }
        A[i][N - 1] = 1.0 - sum;

        sum = 0.0;
        for (int j = 0; j < M - 1; j++) {
            float r = (rand() % 100) / 100000.f;
            if (j % 2 == 0) B[i][j] = (1.0 / M) - r;
            else B[i][j] = (1.0 / M) + r;
            sum += B[i][j];
        }
        B[i][M - 1] = 1.0 - sum;

        float r = (rand() % 100) / 100000.f;
        if (i % 2 == 0) pi[i] = (1.0 / N) - r;
        else pi[i] = (1.0 / N) - r;
    }
    pi[N - 1] = 1.0;
    for (int i = 0; i < N - 1; i++) pi[N - 1] -= pi[i];

    return;
}

double **alpha_pass_no_scale(int N, int T, int *O) {
    // alpha is NxT
    double **alpha = (double **)malloc(sizeof(double *) * N);
    for (int i = 0; i < N; i++) alpha[i] = (double *)malloc(sizeof(double) * T);

    C = (double *)malloc(sizeof(double) * T);

    // compute alpha-pass
    for (int i = 0; i < N; i++) alpha[i][0] = pi[i] * B[i][O[0]];

    // compute alpha_t(i)
    for (int t = 1; t < T; t++) {
        for (int i = 0; i < N; i++) {
            alpha[i][t] = 0;
            for (int j = 0; j < N; j++) alpha[i][t] += (alpha[j][t - 1] * A[j][i]);
            alpha[i][t] *= B[i][O[t]];
        }
    }

    return alpha;
}

double **alpha_pass(int N, int T, int *O) {
    // alpha is NxT
    double **alpha = (double **)malloc(sizeof(double *) * N);
    for (int i = 0; i < N; i++) alpha[i] = (double *)malloc(sizeof(double) * T);

    C = (double *)malloc(sizeof(double) * T);

    // compute alpha-pass
    C[0] = 0.0;
    for (int i = 0; i < N; i++) {
        int q = O[0];
        alpha[i][0] = pi[i] * B[i][q];
        C[0] = C[0] + alpha[i][0];
    }

    // scale the alpha_0(i)
    C[0] = 1.0 / C[0];
    for (int i = 0; i < N; i++) alpha[i][0] *= C[0];

    // compute alpha_t(i)
    for (int t = 1; t < T; t++) {
        C[t] = 0;
        for (int i = 0; i < N; i++) {
            alpha[i][t] = 0;
            for (int j = 0; j < N; j++) {
                alpha[i][t] += (alpha[j][t - 1] * A[j][i]);
            }
            int q = O[t];
            alpha[i][t] *= B[i][q];
            C[t] += alpha[i][t];
        }

        // scale alpha_t(i)
        C[t] = 1.0 / C[t];
        for (int i = 0; i < N; i++) {
            alpha[i][t] *= C[t];
        }
    }

    return alpha;
}

double **beta_pass(int N, int T, int *O) {
    // beta is NxT
    double **beta = (double **)malloc(sizeof(double *) * N);
    for (int i = 0; i < N; i++) beta[i] = (double *)malloc(sizeof(double) * T);

    for (int i = 0; i < N; i++) {
        beta[i][T - 1] = C[T - 1];
    }

    for (int t = T - 2; t >= 0; t--) {
        for (int i = 0; i < N; i++) {
            beta[i][t] = 0.0;
            for (int j = 0; j < N; j++) {
                int q = O[t + 1];
                beta[i][t] += (A[i][j] * B[j][q] * beta[j][t + 1]);
            }
            // scale beta_t(i)
            beta[i][t] *= C[t];
        }
    }

    return beta;
}

double ***compute_gamma(int N, int T, int *O, double **alpha, double **beta, double ***gamma) {

    // gamma: NxT
    *gamma = (double **)malloc(sizeof(double *) * N);
    for (int i = 0; i < N; i++) (*gamma)[i] = (double *)malloc(sizeof(double) * T);

    // // di-gamma: NxNxT
    double ***di_gamma = (double ***)malloc(sizeof(double **) * N);
    for (int i = 0; i < N; i++) {
        di_gamma[i] = (double **)malloc(sizeof(double *) * N);
        for (int j = 0; j < N; j++) di_gamma[i][j] = (double *)malloc(sizeof(double) * T);
    }

    for (int t = 0; t < T - 1; t++) {
        double denom  = 0.0;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                int q = O[t + 1];
                denom += (alpha[i][t] * A[i][j] * B[j][q] * beta[j][t + 1]);
            }
        }

        for (int i = 0; i < N; i++) {
            (*gamma)[i][t] = 0.0;
            for (int j = 0; j < N; j++) {
                int q = O[t + 1];
                di_gamma[i][j][t] = (alpha[i][t] * A[i][j] * B[j][q] * beta[j][t + 1]) / denom;
                (*gamma)[i][t] += di_gamma[i][j][t];
            }
        }
    }

    // special case for gamma_t-1(i)
    double denom = 0.0;
    for (int i = 0; i < N; i++) {
        denom += alpha[i][T - 1];
    }

    for (int i = 0; i < N; i++) {
        (*gamma)[i][T - 1] = alpha[i][T - 1] / denom;
    }

    return di_gamma;
}

void re_estimate(int N, int M, int T, int *O, double **gamma, double ***di_gamma) {
    // re-estimate pi
    for (int i = 0; i < N; i++) pi[i] = gamma[i][0];

    // re-estimate A
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double numer = 0.0;
            double denom = 0.0;
            for (int t = 0; t < T - 1; t++) {
                numer += di_gamma[i][j][t];
                denom += gamma[i][t];
            }
            A[i][j] = numer / denom;
        }
    }

    // re-estimate B
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            double numer = 0.0;
            double denom = 0.0;
            for (int t = 0; t < T; t++) {
                if (O[t] == j) numer += gamma[i][t];
                denom += gamma[i][t];
            }
            B[i][j] = numer / denom;
        }
    }
}

double compute_prob(int T) {
    double log_prob = 0.0;
    for (int i = 0; i < T; i++) {
        log_prob += log(C[i]);
    }
    log_prob = -log_prob;
    return log_prob;
}
