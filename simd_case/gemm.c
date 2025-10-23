// gemm.c
// Compile variants:
// scalar: gcc gemm.c -O2 -o gemm_scalar
// auto:   gcc gemm.c -O3 -march=native -o gemm_auto
// avx2:   gcc gemm.c -O3 -mavx2 -mfma -march=native -DUSE_AVX2 -o gemm_avx2

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#ifdef USE_AVX2
#include <immintrin.h>
#endif

#define N 128
// tile sizes
#define TI 32
#define TK 32
#define TJ 32

static float A[N][N];
static float B[N][N];
static float C[N][N];

void init_data()
{
    srand(0);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            A[i][j] = (float)(rand() % 256) / 255.0f;
            B[i][j] = (float)(rand() % 256) / 255.0f;
            C[i][j] = 0.0f;
        }
}

void gemm_scalar()
{
    for (int ii = 0; ii < N; ii += TI)
        for (int kk = 0; kk < N; kk += TK)
            for (int jj = 0; jj < N; jj += TJ)
                for (int i = ii; i < ii + TI && i < N; i++)
                    for (int k = kk; k < kk + TK && k < N; k++)
                        for (int j = jj; j < jj + TJ && j < N; j++)
                            C[i][j] += A[i][k] * B[k][j];
}

#ifdef USE_AVX2
void gemm_avx2()
{
    const int V = 8;
    for (int ii = 0; ii < N; ii += TI)
        for (int kk = 0; kk < N; kk += TK)
            for (int jj = 0; jj < N; jj += TJ)
                for (int i = ii; i < ii + TI && i < N; i++)
                    for (int k = kk; k < kk + TK && k < N; k++)
                    {
                        __m256 a_broadcast = _mm256_set1_ps(A[i][k]);
                        int j = jj;
                        for (; j <= jj + TJ - V && j <= N - V; j += V)
                        {
                            __m256 bvec = _mm256_loadu_ps(&B[k][j]);
                            __m256 cvec = _mm256_loadu_ps(&C[i][j]);
                            cvec = _mm256_fmadd_ps(a_broadcast, bvec, cvec);
                            _mm256_storeu_ps(&C[i][j], cvec);
                        }
                        for (; j < jj + TJ && j < N; j++)
                        {
                            C[i][j] += A[i][k] * B[k][j];
                        }
                    }
}
#endif

int main()
{
    init_data();
    // warm-up
    for (int w = 0; w < 2; w++)
    {
#ifdef USE_AVX2
        gemm_avx2();
#else
        gemm_scalar();
#endif
    }

    clock_t start = clock();
#ifdef USE_AVX2
    gemm_avx2();
#else
    gemm_scalar();
#endif
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("%.9f\n", elapsed);

    // checksum
    volatile double s = 0;
    for (int i = 0; i < N; i += 7)
        s += C[i][i];
    (void)s;
    return 0;
}
