#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include <string.h>
#include <ctype.h>
#include "mycipher.h"

const int key = 19;

char* gen_cipher_with_shifting(int *cipher_length) {
    char * line = NULL;
    size_t len = 0;
    FILE *fp = fopen("./dataset/brown.txt", "r");
    if (fp == NULL) {fprintf(stderr, "ERROR: Couldn't open file.\n"); exit(1);};

    const int T = 50000;
    *cipher_length = T;
    ssize_t read;
    int count = 0;
    char *cipher = (char *)malloc(sizeof(char) * T);
    while ((read = getline(&line, &len, fp)) != -1){
        for (int i = 0; i < len; i++) {
            char c = line[i];
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
                // if (c != ' ') O[count] = tolower(c) - 'a';
                // else O[count] = 26;
                char lo_c = tolower(c) - 'a';
                lo_c = ((lo_c + key) % 26) + 'a';
                cipher[count] = tolower(lo_c);
                count++;
            }
            if (count >= T) break;
        }
        if (count >= T) break;
    }
    
    return cipher;
}

double** init_english_diagram() {
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

    // compute log probability
    for (int i = 0; i < 26; i++) {
        long long row_sum = 0;
        for (int j = 0; j < 26; j++) row_sum += char_bigram[i][j];
        for (int j = 0; j < 26; j++) char_bigram[i][j] /= row_sum;
    }

    fclose(fp);
    if (line) free(line);

    return char_bigram;
}