// Hidden Markov Model
// Author: WeiChung Huang
// Last Modified on 9/6/2017

#include "stdio.h"
#include "stdlib.h"
#include "time.h"
#include "math.h"
#include <ctype.h>
#include <string.h>

#include "mycipher.h"

void hmm_train(int M, int N, int T, int *O, int max_iter);
void init_model(int M, int N, int update_A);
void dump_model(int M, int N);
void alpha_pass(int N, int T, int *O);
void alpha_pass_no_scale(int N, int T, int *O);
void beta_pass(int N, int T, int *O);
void compute_gamma(int N, int T, int *O);
void re_estimate(int N, int M, int T, int *O);
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

double **alpha = NULL;
double **beta = NULL;
double **gamma_t = NULL;
double ***di_gamma = NULL;

int g_random_seed = 0;

void dump_model(int M, int N) {
    // dump A-matrix
    printf("\n--------A-Matrix--------\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.4f ", A[i][j]);
        }
        printf("\n");
    }

    printf("\n--------B-Matrix--------\n");
    for (int i = 0; i < M; i++) {
        // if (i < M - 1) printf("    %c | ", i + 'a');
        // else printf("space | ");
        printf("    %c | ", i + 'a');
        for (int j = 0; j < N; j++) {
            printf("%.4f ", B[j][i]);
        }
        printf("\n");
    }

    printf("\n--------pi-Matrix--------\n");
    for (int i = 0; i < N; i++) {
        printf("%.4f ", pi[i]);
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
    // init observations
    O = (int *)malloc(sizeof(int) * T);

    char * line = NULL;
    size_t len = 0;
    FILE *fp = fopen("../dataset/brown.txt", "r");
    if (fp == NULL) {fprintf(stderr, "ERROR: Couldn't open file (brown.txt).\n"); exit(1);};

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
    char *cipher = gen_cipher_with_shifting(T);
    
    O = (int *)malloc(sizeof(int) * T);
    for (int i = 0; i < cipher_length; i++) O[i] = cipher[i] - 'a';
    free(cipher);

    double **digraph = init_english_digraph();
    for (int i = 0; i < 26; i++) {
        for (int j = 0; j < 26; j++) A[i][j] = digraph[i][j];
    }
    free(digraph);
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
            alpha_pass_no_scale(N, T, O);
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

void free_global_memory(int N) {
    if (A) {
        for (int i = 0; i < N; i++) free(A[i]);
        free(A); A = NULL;
    }
    if (B) {
        for (int i = 0; i < N; i++) free(B[i]);
        free(B); B = NULL;
    }
    if (pi) {free(pi); pi = NULL;}
    if (O) {free(O); O = NULL;}
    if (C) {free(C); C = NULL;}

    // free memory
    if (alpha) {
        for (int i = 0; i < N; i++) free(alpha[i]);
        free(alpha);
    }

    if (beta) {
        for (int i = 0; i < N; i++) free(beta[i]);
        free(beta);
    }

    if (gamma_t) {
        for (int i = 0; i < N; i++) free(gamma_t[i]);
        free(gamma_t);   
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
}

int main(int argc, char *argv[]) {
    if (argc < 6) {
        fprintf(stderr, "[Usage]: M N T maxIteration numModels seed(optional)\n");
        exit(1);
    }
    const int M = atoi(argv[1]);
    const int N = atoi(argv[2]);
    const int T = atoi(argv[3]);
    const int max_iter = atoi(argv[4]);
    const int num_models = atoi(argv[5]);
    g_random_seed = time(NULL);
    if (argc > 6) g_random_seed = atoi(argv[6]);

    float max_score = 0.f;
    init_model(M, N, 1);
    init_ss_cipher_corpus(T);
    for (int m = 0; m < num_models; m++) {
        hmm_train(M, N, T, O, max_iter);
        extract_putative_key(B);
        float score = score_cipher(T);
        init_model(M, N, 0);
        g_random_seed = rand() % (1000000);
    }

    // init_ss_cipher_corpus(T);
    // Config: M = 26, N = 26, T = 1000

    // init_simple_corpus(T);
    // // Config: M = 3, N = 2, T = 4

    // init_brown_corpus(T);
    // // Config: M = 27, N = 2~, T = 50000, max_iter = 500

    // hmm_train(M, N, T, O, max_iter);

    
    // float score = score_cipher(T);
    // printf("score: %f\n", score);
    // printf("Fraction of putative key: %f and max score: %f\n", score, max_score);
    // if (score > max_score) max_score = score;

    // printf("MAX SCORE: %f\n", max_score);
    free_global_memory(N);
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
        alpha_pass(N, T, O);
        beta_pass(N, T, O);
        compute_gamma(N, T, O);
        re_estimate(N, M, T, O);
        double new_prob = compute_prob(T);        
        old_prob = new_prob;
        iteration++;

        if (max_iter > 10 && (iteration % (int)(max_iter / 10.f)) == 0) {
            printf("----------------\n");
            printf("iter: %d(%.0f%%), prob: %f\n", iteration, iteration * 100.f / (float)max_iter, new_prob);
        }
    }
    dump_model(M, N);
}

void init_model(int M, int N, int update_A) {
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
    srand(g_random_seed);
    const float precision = 1.f / 10.f;
    float cell[2] = {1.0 / N, 1.0 / M};
    float max_cell[2] = {0.f};
    float min_cell[2] = {0.f};
    float cell_range[2] = {0.f};
    for (int i = 0; i < 2; i++) {
        max_cell[i] = cell[i] + cell[i] * precision;
        min_cell[i] = cell[i] - cell[i] * precision;
        cell_range[i] = max_cell[i] - min_cell[i];
    }
    for (int i = 0; i < N; i++) {
        double sum = 0.0;
        if (update_A) {
            for (int j = 0; j < N - 1; j++) {
                float r = (rand() % 1000) / 1000.f;
                A[i][j] = min_cell[0] + cell_range[0] * r;
                sum += A[i][j];
            }
        }
        A[i][N - 1] = 1.0 - sum;

        sum = 0.0;
        for (int j = 0; j < M - 1; j++) {
            float r = (rand() % 1000) / 1000.f;
            B[i][j] = min_cell[1] + cell_range[1] * r;
            sum += B[i][j];
        }
        B[i][M - 1] = 1.0 - sum;

        float r = (rand() % 1000) / 1000.f;
        pi[i] = min_cell[0] + cell_range[0] * r;
    }
    pi[N - 1] = 1.0;
    for (int i = 0; i < N - 1; i++) pi[N - 1] -= pi[i];

    return;
}

void alpha_pass_no_scale(int N, int T, int *O) {
    // alpha is NxT
    if (!alpha) {
        alpha = (double **)malloc(sizeof(double *) * N);
        for (int i = 0; i < N; i++) alpha[i] = (double *)malloc(sizeof(double) * T);
    }

    if (!C) C = (double *)malloc(sizeof(double) * T);

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
}

void alpha_pass(int N, int T, int *O) {
    // alpha is NxT
    if (!alpha) {
        alpha = (double **)malloc(sizeof(double *) * N);
        for (int i = 0; i < N; i++) alpha[i] = (double *)malloc(sizeof(double) * T);
    }

    if (!C) C = (double *)malloc(sizeof(double) * T);

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
}

void beta_pass(int N, int T, int *O) {
    // beta is NxT
    if (!beta) {
        beta = (double **)malloc(sizeof(double *) * N);
        for (int i = 0; i < N; i++) beta[i] = (double *)malloc(sizeof(double) * T);
    }

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
}

void compute_gamma(int N, int T, int *O) {
    // gamma: NxT
    if (!gamma_t) {
        gamma_t = (double **)malloc(sizeof(double *) * N);
        for (int i = 0; i < N; i++) gamma_t[i] = (double *)malloc(sizeof(double) * T);
    }

    // // di-gamma: NxNxT
    if (!di_gamma) {
        di_gamma = (double ***)malloc(sizeof(double **) * N);
        for (int i = 0; i < N; i++) {
            di_gamma[i] = (double **)malloc(sizeof(double *) * N);
            for (int j = 0; j < N; j++) di_gamma[i][j] = (double *)malloc(sizeof(double) * T);
        }
    }

    for (int t = 0; t < T - 1; t++) {
        double denom  = 0.0;
        for (int i = 0; i < N; i++) {
#if 1 // they are should be equalvent. For performance concern, use faster one
            denom += alpha[i][T - 1];
#else
            for (int j = 0; j < N; j++) {
                int q = O[t + 1];
                denom += (alpha[i][t] * A[i][j] * B[j][q] * beta[j][t + 1]);
            }
#endif
        }

        for (int i = 0; i < N; i++) {
            gamma_t[i][t] = 0.0;
            for (int j = 0; j < N; j++) {
                int q = O[t + 1];
                di_gamma[i][j][t] = (alpha[i][t] * A[i][j] * B[j][q] * beta[j][t + 1]) / denom;
                gamma_t[i][t] += di_gamma[i][j][t];
            }
        }
    }

    // special case for gamma_t-1(i)
    double denom = 0.0;
    for (int i = 0; i < N; i++) {
        denom += alpha[i][T - 1];
    }

    for (int i = 0; i < N; i++) {
        gamma_t[i][T - 1] = alpha[i][T - 1] / denom;
    }
}

void re_estimate(int N, int M, int T, int *O) {
    // re-estimate pi
    for (int i = 0; i < N; i++) pi[i] = gamma_t[i][0];

    // // re-estimate A
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         double numer = 0.0;
    //         double denom = 0.0;
    //         for (int t = 0; t < T - 1; t++) {
    //             numer += di_gamma[i][j][t];
    //             denom += gamma_t[i][t];
    //         }
    //         A[i][j] = numer / denom;
    //     }
    // }

    // re-estimate B
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            double numer = 0.0;
            double denom = 0.0;
            for (int t = 0; t < T; t++) {
                if (O[t] == j) numer += gamma_t[i][t];
                denom += gamma_t[i][t];
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
