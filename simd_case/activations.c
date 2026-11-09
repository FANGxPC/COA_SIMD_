// activations.c
// Compile variants similarly with -DUSE_AVX2 for AVX2 variant.


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

#ifdef USE_AVX2
#include <immintrin.h>
#endif

#define LEN 1 << 20 // about 1M elements

static float X[LEN];
static float Y[LEN];

void init_data()
{
    srand(0);
    for (int i = 0; i < LEN; i++)
        X[i] = ((float)(rand() % 512) - 256.0f) / 64.0f; // some negative and positive
}

void relu_scalar()
{
    for (int i = 0; i < LEN; i++)
        Y[i] = X[i] > 0.0f ? X[i] : 0.0f;
}

void sigmoid_scalar()
{
    for (int i = 0; i < LEN; i++)
        Y[i] = 1.0f / (1.0f + expf(-X[i]));
}

#ifdef USE_AVX2
void relu_avx2()
{
    const int V = 8;
    __m256 zero = _mm256_setzero_ps();
    int i = 0;
    for (; i <= LEN - V; i += V)
    {
        __m256 xv = _mm256_loadu_ps(&X[i]);
        __m256 yv = _mm256_max_ps(xv, zero);
        _mm256_storeu_ps(&Y[i], yv);
    }
    for (; i < LEN; i++)
        Y[i] = X[i] > 0.0f ? X[i] : 0.0f;
}

// Sigmoid vectorized properly is complex (requires vector exp). For correctness we keep scalar sigmoid.
#endif

int main(int argc, char **argv)
{
    init_data();
    // Choose operation by env var OP:
    // "relu" or "sigmoid"
    const char *op = getenv("OP");
    if (!op)
        op = "relu";

    for (int w = 0; w < 3; w++)
    {
#ifdef USE_AVX2
        if (strcmp(op, "relu") == 0)
            relu_avx2();
        else
            sigmoid_scalar();
#else
        if (strcmp(op, "relu") == 0)
            relu_scalar();
        else
            sigmoid_scalar();
#endif
    }

    clock_t start = clock();
#ifdef USE_AVX2
    if (strcmp(op, "relu") == 0)
        relu_avx2();
    else
        sigmoid_scalar();
#else
    if (strcmp(op, "relu") == 0)
        relu_scalar();
    else
        sigmoid_scalar();
#endif
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("%.9f\n", elapsed);
    // checksum
    volatile double s = 0;
    for (int i = 0; i < LEN; i += 123457)
        s += Y[i];
    (void)s;
    return 0;
}
