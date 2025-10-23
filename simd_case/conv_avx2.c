// conv_avx2.c - 3x3 convolution using AVX2 intrinsics

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>

#define H 512
#define W 512

float img[H][W];
float out[H][W];
float k[3][3] = {
    {0, 1, 0},
    {1, -4, 1},
    {0, 1, 0}};

void conv3x3_avx2()
{
    __m256 krow0 = _mm256_set_ps(0, 1, 0, 0, 1, 0, 0, 1);
    __m256 krow1 = _mm256_set_ps(1, -4, 1, 1, -4, 1, 1, -4);
    __m256 krow2 = _mm256_set_ps(0, 1, 0, 0, 1, 0, 0, 1);

    for (int i = 1; i < H - 1; i++)
    {
        for (int j = 1; j < W - 9; j += 8)
        {
            __m256 sum = _mm256_setzero_ps();

            __m256 top = _mm256_loadu_ps(&img[i - 1][j]);
            __m256 mid = _mm256_loadu_ps(&img[i][j]);
            __m256 bot = _mm256_loadu_ps(&img[i + 1][j]);

            sum = _mm256_fmadd_ps(top, krow0, sum);
            sum = _mm256_fmadd_ps(mid, krow1, sum);
            sum = _mm256_fmadd_ps(bot, krow2, sum);

            _mm256_storeu_ps(&out[i][j], sum);
        }
    }
}

int main()
{
    srand(0);
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++)
            img[i][j] = rand() % 256;

    clock_t start = clock();
    conv3x3_avx2();
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("%f\n", elapsed);
    return 0;
}
