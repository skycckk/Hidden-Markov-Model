

char* gen_cipher_with_shifting(int cipher_length);
double** init_english_digraph();
void extract_putative_key(double **B, int N, int M);
float score_cipher(int text_len);
int* read_zodiac408(const char *file);
void dealloc_cipher();