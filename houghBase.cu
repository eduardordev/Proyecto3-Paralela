/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   :
 To build use  : make
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include "pgm.h"

const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;

void CPU_HoughTran(unsigned char *pic, int w, int h, int **acc) {
    float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
    *acc = new int[rBins * degreeBins];
    memset(*acc, 0, sizeof(int) * rBins * degreeBins);
    int xCent = w / 2;
    int yCent = h / 2;
    float rScale = 2 * rMax / rBins;

    for (int i = 0; i < w; i++)
        for (int j = 0; j < h; j++) {
            int idx = j * w + i;
            if (pic[idx] > 0) {
                int xCoord = i - xCent;
                int yCoord = yCent - j;
                float theta = 0;
                for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
                    float r = xCoord * cos(theta) + yCoord * sin(theta);
                    int rIdx = (r + rMax) / rScale;
                    (*acc)[rIdx * degreeBins + tIdx]++;
                    theta += radInc;
                }
            }
        }
}

__global__ void GPU_HoughTran(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale, float *d_Cos, float *d_Sin) {
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if (gloID >= w * h) return;

    int xCent = w / 2;
    int yCent = h / 2;
    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;

    if (pic[gloID] > 0) {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
            int rIdx = (r + rMax) / rScale;
            atomicAdd(acc + (rIdx * degreeBins + tIdx), 1);
        }
    }
}

int main(int argc, char **argv) {
    PGMImage inImg(argv[1]);
    int *cpuht;
    int w = inImg.x_dim;
    int h = inImg.y_dim;

    float* d_Cos;
    float* d_Sin;
    cudaMalloc((void **) &d_Cos, sizeof(float) * degreeBins);
    cudaMalloc((void **) &d_Sin, sizeof(float) * degreeBins);

    CPU_HoughTran(inImg.pixels, w, h, &cpuht);

    float *pcCos = (float *) malloc(sizeof(float) * degreeBins);
    float *pcSin = (float *) malloc(sizeof(float) * degreeBins);
    float rad = 0;
    for (int i = 0; i < degreeBins; i++) {
        pcCos[i] = cos(rad);
        pcSin[i] = sin(rad);
        rad += radInc;
    }

    float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
    float rScale = 2 * rMax / rBins;
    cudaMemcpy(d_Cos, pcCos, sizeof(float) * degreeBins, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sin, pcSin, sizeof(float) * degreeBins, cudaMemcpyHostToDevice);

    unsigned char *d_in, *h_in;
    int *d_hough, *h_hough;
    h_in = inImg.pixels;
    h_hough = (int *) malloc(degreeBins * rBins * sizeof(int));

    cudaMalloc((void **) &d_in, sizeof(unsigned char) * w * h);
    cudaMalloc((void **) &d_hough, sizeof(int) * degreeBins * rBins);
    cudaMemcpy(d_in, h_in, sizeof(unsigned char) * w * h, cudaMemcpyHostToDevice);
    cudaMemset(d_hough, 0, sizeof(int) * degreeBins * rBins);

    int blockNum = ceil(w * h / 256.0);

    // Crear los eventos paso 2
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


    GPU_HoughTran<<<blockNum, 256>>>(d_in, w, h, d_hough, rMax, rScale, d_Cos, d_Sin);
    // Liberar los eventos paso 2 cambiar para que este luego de la barrera de sincronizacion
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Tiempo de ejecución del kernel: %f sec\n", milliseconds);

    cudaMemcpy(h_hough, d_hough, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

    for (int i = 0; i < degreeBins * rBins; i++) {
        if (cpuht[i] != h_hough[i])
            printf("Calculation mismatch at: %i %i %i\n", i, cpuht[i], h_hough[i]);
    }
    printf("Done!\n");
    // Liberar los eventos paso 2
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Liberar la memoria asignada
    cudaFree(d_Cos);
    cudaFree(d_Sin);
    cudaFree(d_in);
    cudaFree(d_hough);

    delete[] cpuht;
    free(pcCos);
    free(pcSin);
    free(h_hough);

    return 0;
}
