#include <stdio.h>
#include <cuda.h>

#define N 1024  // You can tune this

__global__ void gemv_gpu(const float *A, const float *x, float *y, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        float sum = 0.0f;
        for (int j = 0; j < n; ++j)
            sum += A[row * n + j] * x[j];
        y[row] = sum;
    }
}

int main() {
    int size_A = N * N * sizeof(float);
    int size_x = N * sizeof(float);
    int size_y = N * sizeof(float);

    float *h_A = (float*)malloc(size_A);
    float *h_x = (float*)malloc(size_x);
    float *h_y = (float*)malloc(size_y);

    for (int i = 0; i < N * N; ++i) h_A[i] = 1.0f;
    for (int i = 0; i < N; ++i) h_x[i] = 1.0f;

    float *d_A, *d_x, *d_y;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_x, size_x);
    cudaMalloc(&d_y, size_y);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, size_x, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    gemv_gpu<<<blocks, threads>>>(d_A, d_x, d_y, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_y, d_y, size_y, cudaMemcpyDeviceToHost);

    printf("GPU GEMV done (N=%d)\n", N);

    cudaFree(d_A); cudaFree(d_x); cudaFree(d_y);
    free(h_A); free(h_x); free(h_y);
    return 0;
}

