// conv_auto.c - 3x3 convolution (auto-vectorized)

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define H 512
#define W 512

float img[H][W];
float out[H][W];
float k[3][3] = {
    {0, 1, 0},
    {1, -4, 1},
    {0, 1, 0}};

void conv3x3_auto()
{
    for (int i = 1; i < H - 1; i++)
    {
        for (int j = 1; j < W - 1; j++)
        {
            float sum = 0.0f;
            for (int m = 0; m < 3; m++)
            {
                for (int n = 0; n < 3; n++)
                {
                    sum += k[m][n] * img[i + m - 1][j + n - 1];
                }
            }
            out[i][j] = sum;
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
    conv3x3_auto();
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("%f\n", elapsed);
    return 0;
}
