#include <stdio.h>
#include <cuda.h>

__global__ void hello(int totalThreads)
{
    int myID = blockIdx.x * blockDim.x + threadIdx.x;
    int globalID = myID + blockIdx.x * blockDim.x * gridDim.x;

    if (myID < totalThreads)
    {
        if (myID == totalThreads - 1)
        {
            printf("Hello world from the thread with the maximum global ID: %i (global ID: %i)\\n", myID, globalID);
        }
    }
}

int main()
{
    int totalThreads = 100000;  // Número total de hilos
    int threadsPerBlock = 256;  // Número de hilos por bloque
    int numBlocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock; // Asegúrese de lanzar suficientes bloques

    // Llama al kernel con la configuración de bloques y hilos
    hello<<<numBlocks, threadsPerBlock>>>(totalThreads);
    cudaDeviceSynchronize();  // Espera a que finalicen todos los hilos

    return 0;
}
