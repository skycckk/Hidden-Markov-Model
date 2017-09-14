// Some cryptographic utilities
// Author: WeiChung Huang
// Last Modified on 9/13/2017

#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include <string.h>
#include <ctype.h>
#include "mycipher.h"

const int key = 1;

char *g_plaintext = NULL;
char *g_ciphertext = NULL;
char *g_putative_key = NULL;

void dealloc_cipher() {
    if (g_plaintext) {free(g_plaintext); g_plaintext = NULL;}
    if (g_ciphertext) {free(g_ciphertext); g_ciphertext = NULL;}
    if (g_putative_key) {free(g_putative_key); g_putative_key = NULL;}
}

char* gen_cipher_with_shifting(int cipher_length) {
    char * line = NULL;
    size_t len = 0;
    FILE *fp = fopen("./dataset/brown_copy.txt", "r");
    if (fp == NULL) {fprintf(stderr, "ERROR: Couldn't open file.\n"); exit(1);};

    int T = cipher_length;
    ssize_t read;
    int count = 0;

    char *cipher = (char *)malloc(sizeof(char) * T);
    if (g_ciphertext) {free(g_ciphertext); g_ciphertext = NULL;}
    g_ciphertext = (char *)malloc(sizeof(char) * T);
    if (g_plaintext) {free(g_plaintext); g_plaintext = NULL;}
    g_plaintext = (char *)malloc(sizeof(char) * T);
    while ((read = getline(&line, &len, fp)) != -1){
        for (int i = 0; i < len; i++) {
            char c = line[i];
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
                g_plaintext[count] = tolower(c);
                char lo_c = tolower(c) - 'a';
                lo_c = ((lo_c + key) % 26) + 'a';
                cipher[count] = tolower(lo_c);
                g_ciphertext[count] = tolower(lo_c);
                count++;
            }
            if (count >= T) break;
        }
        if (count >= T) break;
    }
    
    return cipher;
}

int* read_zodiac408(const char* file) {
    char * line = NULL;
    size_t len = 0;
    FILE *fp = fopen(file, "r");
    if (fp == NULL) {fprintf(stderr, "ERROR: Couldn't open file. (%s)\n", file); exit(1);};

    const int T = 408;
    ssize_t read;
    int count = 0;
    char delimit[] = " \t\r\n\v\f";
    int *cipher = (int *)malloc(sizeof(int) * T);

    if (g_ciphertext) {free(g_ciphertext); g_ciphertext = NULL;}
    g_ciphertext = (char *)malloc(sizeof(char) * T);
    if (g_plaintext) {free(g_plaintext); g_plaintext = NULL;}
    g_plaintext = (char *)malloc(sizeof(char) * T);
    while ((read = getline(&line, &len, fp)) != -1) {
        char *token = strtok(line, delimit);
        while (token != NULL) {
            cipher[count] = atoi(token) - 1;
            g_ciphertext[count] = cipher[count];
            count++;
            token = strtok(NULL, delimit);
        }
    }
    fclose(fp);

    // read the plaintext as well
    fp = fopen("./dataset/zodiac408_plaintext.txt", "r");
    if (fp == NULL) {fprintf(stderr, "ERROR: Couldn't open file. (zodiac408_plain)\n"); exit(1);};
    count = 0;
    while ((read = getline(&line, &len, fp)) != -1) {
        char *token = strtok(line, delimit);
        while (token != NULL) {
            g_plaintext[count] = token[0];
            count++;
            token = strtok(NULL, delimit);
        }
    }
    fclose(fp);

    return cipher;
}

double** init_english_digraph() {
    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;
    const char *file_path = "./dataset/english_bigrams.txt";

    fp = fopen(file_path, "r");
    if (fp == NULL) {fprintf(stderr, "ERROR: Couldn't open file: %s\n", file_path); exit(1);};
    double **char_bigram = (double **)malloc(sizeof(double *) * 26);
    for (int i = 0; i < 26; i++) char_bigram[i] = (double *)malloc(sizeof(double) * 26);

    while ((read = getline(&line, &len, fp)) != -1) {
        char *pch = strtok(line, " ");
        char *bi = pch;
        
        int bi_index0 = toupper(bi[0]) - 'A';
        int bi_index1 = toupper(bi[1]) - 'A';

        pch = strtok(NULL, " ");
        int count = atoi(pch);

        char_bigram[bi_index0][bi_index1] = count;
    }

    // compute probability
    for (int i = 0; i < 26; i++) {
        long long row_sum = 0.0;
        for (int j = 0; j < 26; j++) {char_bigram[i][j] += 5; row_sum += char_bigram[i][j];};
        for (int j = 0; j < 26; j++) char_bigram[i][j] /= row_sum;
    }

    fclose(fp);
    if (line) free(line);

    return char_bigram;
}

void extract_putative_key(double **B, int N, int M) {
    if (!B) return;
    if (!g_putative_key) g_putative_key = (char *)malloc(sizeof(char) * M);
    for (int j = 0; j < M; j++) {
        double max_prob = 0.0;
        int hidden_letter = 0;
        for (int i = 0; i < N; i++) {
            if (B[i][j] >= max_prob) {
                max_prob = B[i][j];
                hidden_letter = i;
            }
        }
        // the mapping would be j -> hidden_letter with max prob.
        g_putative_key[j] = hidden_letter + 'a';
    }
}

float score_cipher(int text_len) {
    int count = 0;
    for (int i = 0; i < text_len; i++) {
        //if ((g_putative_key[i] - 'a') == ((i - key) % 26)) count++;
        int idx = g_ciphertext[i];
        if (g_putative_key[idx] == g_plaintext[i]) count++;
    }

    return count / (float)text_len;
}
