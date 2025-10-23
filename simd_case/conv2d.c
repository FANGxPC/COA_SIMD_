// conv2d.c
// scalar: gcc conv2d.c -O2 -o conv2d_scalar
// auto:   gcc conv2d.c -O3 -march=native -o conv2d_auto
// avx2:   gcc conv2d.c -O3 -mavx2 -mfma -march=native -DUSE_AVX2 -o conv2d_avx2

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#ifdef USE_AVX2
#include <immintrin.h>
#endif

#define H 512
#define W 512

static float img[H][W];
static float out[H][W];
// example gaussian kernel normalized
static const float k[3][3] = {
    {1 / 16.0f, 2 / 16.0f, 1 / 16.0f},
    {2 / 16.0f, 4 / 16.0f, 2 / 16.0f},
    {1 / 16.0f, 2 / 16.0f, 1 / 16.0f}};

void init_data()
{
    srand(0);
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++)
            img[i][j] = (float)(rand() % 256) / 255.0f;
    memset(out, 0, sizeof(out));
}

void conv2d_scalar()
{
    for (int i = 1; i < H - 1; i++)
    {
        for (int j = 1; j < W - 1; j++)
        {
            float s = 0.0f;
            for (int m = 0; m < 3; m++)
                for (int n = 0; n < 3; n++)
                    s += k[m][n] * img[i + m - 1][j + n - 1];
            out[i][j] = s;
        }
    }
}

#ifdef USE_AVX2
void conv2d_avx2()
{
    // We vectorize across the width (j) with 8-wide floats
    const int V = 8;
    __m256 k0 = _mm256_set1_ps(k[0][1]); // center of top row
    // Instead of setting each coefficient separately in fancy way (for clarity use loads & scalar mul-add)
    for (int i = 1; i < H - 1; i++)
    {
        int j = 1;
        for (; j <= W - 1 - V; j += V)
        {
            __m256 sum = _mm256_setzero_ps();
            // top row
            __m256 t0 = _mm256_loadu_ps(&img[i - 1][j - 1]); // j-1 .. j+6
            __m256 t1 = _mm256_loadu_ps(&img[i - 1][j]);     // j .. j+7
            __m256 t2 = _mm256_loadu_ps(&img[i - 1][j + 1]); // j+1 .. j+8
            // mid row
            __m256 m0 = _mm256_loadu_ps(&img[i][j - 1]);
            __m256 m1 = _mm256_loadu_ps(&img[i][j]);
            __m256 m2 = _mm256_loadu_ps(&img[i][j + 1]);
            // bot row
            __m256 b0 = _mm256_loadu_ps(&img[i + 1][j - 1]);
            __m256 b1 = _mm256_loadu_ps(&img[i + 1][j]);
            __m256 b2 = _mm256_loadu_ps(&img[i + 1][j + 1]);
            // accumulate with constants
            sum = _mm256_fmadd_ps(t0, _mm256_set1_ps(k[0][0]), sum);
            sum = _mm256_fmadd_ps(t1, _mm256_set1_ps(k[0][1]), sum);
            sum = _mm256_fmadd_ps(t2, _mm256_set1_ps(k[0][2]), sum);

            sum = _mm256_fmadd_ps(m0, _mm256_set1_ps(k[1][0]), sum);
            sum = _mm256_fmadd_ps(m1, _mm256_set1_ps(k[1][1]), sum);
            sum = _mm256_fmadd_ps(m2, _mm256_set1_ps(k[1][2]), sum);

            sum = _mm256_fmadd_ps(b0, _mm256_set1_ps(k[2][0]), sum);
            sum = _mm256_fmadd_ps(b1, _mm256_set1_ps(k[2][1]), sum);
            sum = _mm256_fmadd_ps(b2, _mm256_set1_ps(k[2][2]), sum);

            _mm256_storeu_ps(&out[i][j], sum);
        }
        // tail
        for (; j < W - 1; j++)
        {
            float s = 0.0f;
            for (int m = 0; m < 3; m++)
                for (int n = 0; n < 3; n++)
                    s += k[m][n] * img[i + m - 1][j + n - 1];
            out[i][j] = s;
        }
    }
}
#endif

int main()
{
    init_data();
    for (int w = 0; w < 3; w++)
    {
#ifdef USE_AVX2
        conv2d_avx2();
#else
        conv2d_scalar();
#endif
    }
    clock_t start = clock();
#ifdef USE_AVX2
    conv2d_avx2();
#else
    conv2d_scalar();
#endif
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("%.9f\n", elapsed);
    // checksum
    volatile double s = 0;
    for (int i = 0; i < H; i += 31)
        s += out[i][i];
    (void)s;
    return 0;
}
