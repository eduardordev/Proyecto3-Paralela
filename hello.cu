#include <stdio.h>

__global__ void helloCUDA()
{
    int threadID = threadIdx.x; // Modifica la forma en que se calcula el identificador del hilo
    printf("Hello, World! I'm thread %d. Nuestro nombre es Jorge Caballeros, Ale Guzman y Eduardo Ramirez\n", threadID);
}

int main()
{
    int numBlocks = 1;          // Número de bloques (1 bloque)
    int threadsPerBlock = 1024; // Número de hilos por bloque (2048 hilos)

    // Llama al kernel con la configuración de bloques y hilos
    helloCUDA<<<numBlocks, threadsPerBlock>>>();
    helloCUDA<<<numBlocks, threadsPerBlock>>>();

    // Espera a que todos los hilos hayan terminado
    cudaDeviceSynchronize();

    return 0;
}
