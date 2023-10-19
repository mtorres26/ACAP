#include <stdio.h>

__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x; // ID global del hilo
  if (i < n) y[i] = a*x[i] + y[i]; // Cada hilo hace una operacion de multiplicacion y suma
}

int main(void)
{
    int N = 20 * (1 << 20); // 20 * 2^20
    float *x, *y, *d_x, *d_y;
    x = (float*)malloc(N*sizeof(float));
    y = (float*)malloc(N*sizeof(float));

    cudaMalloc(&d_x, N*sizeof(float)); 
    cudaMalloc(&d_y, N*sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    } 

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

    // Perform SAXPY on 1M elements
    cudaEventRecord(start);
    saxpy<<<(N+511)/512, 512>>>(N, 2.0f, d_x, d_y);
    cudaEventRecord(stop);

    cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop); // Se bloquea CPU hasta que se haga cudaEventRecord(stop)
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float maxError = 0.0f;
    for (int i = 0; i < N; i++){
        maxError = max(maxError, abs(y[i]-4.0f)); //y[i] debe de ser 4 si esta bien hecho
    }
    
    printf("Max error: %f\n", maxError);
    printf("Effective Bandwidth (GB/s): %f\n", N*4*3/milliseconds/1e6); // N*4 es el num de bytes transferidos por array leido
                                                                        // o escrito, el 3 representa la lectura de x y la
                                                                        // lectura y escritura de y
                                                                        // 1e6 es para GB
    printf("GFLOP/s Effective: %f\n", (2*N)/(milliseconds*1e6) ); // t(segundos) * 10^9, pero en ms es *10^6
                                                                // N es el num de elementos del SAXPY, 2N porque
                                                                // cada elemento hace SAXPY hace una operacion de multiply-add
                                                                // que son 2 FLOPs

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);
}