#include "stdio.h"
#include "stdlib.h"
#include <ctype.h>
#include "mycipher.h"

const int key = 19;

char* gen_cipher_with_shifting(int *cipher_length) {
    char * line = NULL;
    size_t len = 0;
    FILE *fp = fopen("./dataset/brown.txt", "r");
    if (fp == NULL) {fprintf(stderr, "ERROR: Couldn't open file.\n"); exit(1);};

    const int T = 20;
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
    
    for (int i = 0; i < T; i++) printf("%c", cipher[i]);
    return cipher;
}