// gemv.c
// Compile variants:
// scalar: gcc gemv.c -O2 -o gemv_scalar
// auto:   gcc gemv.c -O3 -march=native -o gemv_auto
// avx2:   gcc gemv.c -O3 -mavx2 -mfma -march=native -DUSE_AVX2 -o gemv_avx2

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stdint.h>

#ifdef USE_AVX2
#include <immintrin.h>
#endif

// default sizes (tweak if you want)
#define M 128
#define N 128

static float A[M][N];
static float x[N];
static float y[M];

void init_data()
{
    srand(0);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = (float)(rand() % 256) / 255.0f;
    for (int j = 0; j < N; j++)
        x[j] = (float)(rand() % 256) / 255.0f;
    memset(y, 0, sizeof(y));
}

void gemv_scalar()
{
    for (int i = 0; i < M; i++)
    {
        float sum = 0.0f;
        for (int j = 0; j < N; j++)
            sum += A[i][j] * x[j];
        y[i] = sum;
    }
}

#ifdef USE_AVX2
void gemv_avx2()
{
    const int V = 8; // 8 floats per __m256
    for (int i = 0; i < M; i++)
    {
        __m256 acc = _mm256_setzero_ps();
        int j = 0;
        for (; j <= N - V; j += V)
        {
            __m256 a = _mm256_loadu_ps(&A[i][j]);
            __m256 xv = _mm256_loadu_ps(&x[j]);
            acc = _mm256_fmadd_ps(a, xv, acc);
        }
        float tmp[V];
        _mm256_storeu_ps(tmp, acc);
        float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
        for (; j < N; j++)
            sum += A[i][j] * x[j];
        y[i] = sum;
    }
}
#endif

int main()
{
    init_data();
    // warm-up
    for (int w = 0; w < 3; w++)
    {
#ifdef USE_AVX2
        gemv_avx2();
#else
        gemv_scalar();
#endif
    }

    clock_t start = clock();
#ifdef USE_AVX2
    gemv_avx2();
#else
    gemv_scalar();
#endif
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("%.9f\n", elapsed);
    // small checksum to avoid optimizing out
    volatile double s = 0;
    for (int i = 0; i < M; i++)
        s += y[i];
    (void)s;
    return 0;
}
