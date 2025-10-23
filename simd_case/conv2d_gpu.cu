#include <stdio.h>
#include <cuda.h>

#define N 512   // image size
#define K 3     // kernel size

__global__ void conv2d_gpu(const float *img, const float *kernel, float *out, int n, int k) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int half = k / 2;
    if (x >= half && x < n - half && y >= half && y < n - half) {
        float sum = 0.0f;
        for (int i = -half; i <= half; ++i)
            for (int j = -half; j <= half; ++j)
                sum += img[(y + i) * n + (x + j)] * kernel[(i + half) * k + (j + half)];
        out[y * n + x] = sum;
    }
}

int main() {
    int img_size = N * N * sizeof(float);
    int ker_size = K * K * sizeof(float);

    float *h_img = (float*)malloc(img_size);
    float *h_ker = (float*)malloc(ker_size);
    float *h_out = (float*)malloc(img_size);

    for (int i = 0; i < N * N; ++i) h_img[i] = 1.0f;
    for (int i = 0; i < K * K; ++i) h_ker[i] = 1.0f / (K * K);

    float *d_img, *d_ker, *d_out;
    cudaMalloc(&d_img, img_size);
    cudaMalloc(&d_ker, ker_size);
    cudaMalloc(&d_out, img_size);

    cudaMemcpy(d_img, h_img, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ker, h_ker, ker_size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (N + 15) / 16);
    conv2d_gpu<<<blocks, threads>>>(d_img, d_ker, d_out, N, K);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, img_size, cudaMemcpyDeviceToHost);
    printf("GPU Conv2D done (N=%d, K=%d)\n", N, K);

    cudaFree(d_img); cudaFree(d_ker); cudaFree(d_out);
    free(h_img); free(h_ker); free(h_out);
    return 0;
}
