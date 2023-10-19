#include <stdio.h>
#include <ctime>
#include <cuda_runtime.h>

__global__ void maximo(double* numeros, double* buffer) {
    int tId = threadIdx.x + blockIdx.x * blockDim.x;
    int localId = threadIdx.x;
    
    extern __shared__ double partialVec[];
    // extern sirve para reservar memoria dinamica del tamaño que proporcionemos en la llamada al kernel
    
    partialVec[localId] = numeros[tId]; // Copiamos los valores del vector original
    __syncthreads();

    int j = blockDim.x/2;
    while(j != 0){ // Reducimos el vector a la mitad en cada iteracion y el maximo se quedara en la posicion 0
        if(localId < j){ 
            partialVec[localId] = fmax(partialVec[localId], partialVec[localId + j]);
        }
        __syncthreads();
        j /= 2;
    }

    if(localId == 0){ // El hilo 0 de cada bloque mete en buffer el maximo hallado
        buffer[blockIdx.x] = partialVec[0];
    }
}

int main(int argc, char* argv[])
{
    if(argc != 3 || atoi(argv[1]) % atoi(argv[2]) != 0 ||
             ceil(log2(atof(argv[2]))) != floor(log2(atof(argv[2]))) ){
        printf("Se necesitan 2 parámetros:\n-Tamaño de vector.\n-Tamaño de bloque.\nEl primero tiene que ser multiplo del segundo y el segundo debe ser potencia de 2.\n");
        exit(1);
    }
    int vectorTam = atoi(argv[1]);
    int blockTam = atoi(argv[2]);
    int gridTam = vectorTam / blockTam;

    double *x, *d_x, *x_max, *d_x_max, *d_buffer;
    
    //alojar memoria
    x = (double*)malloc(vectorTam*sizeof(double));
    x_max = (double*)malloc(sizeof(double));
    cudaMalloc((void**)&d_x, vectorTam * sizeof(double));
    cudaMalloc((void**)&d_x_max, sizeof(double));
    cudaMalloc((void**)&d_buffer, gridTam * sizeof(double));
    
    unsigned shared_mem_size = blockTam * sizeof(double);
    // memoria dinamica asignada al extern __shared__ array del kernel (para cada bloque)

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for(int i=0; i<vectorTam; i++){
        x[i] = rand()*M_PI/5.0; // Para que tenga decimales
    }

    cudaEventRecord(start);

    cudaMemcpy(d_x, x, vectorTam*sizeof(double), cudaMemcpyHostToDevice);

    maximo<<<gridTam, blockTam, shared_mem_size>>>(d_x, d_buffer);
    maximo<<<1, blockTam, shared_mem_size>>>(d_buffer, d_x_max);

    cudaDeviceSynchronize();
    
    cudaMemcpy(x_max, d_x_max, sizeof(double), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop); // barrera

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Maximo elemento GPU: %lf\n", *x_max);
    printf("Tiempo GPU: %f ms\n", milliseconds);

    cudaEventRecord(start);
    
    *x_max = 0.0;
    for(int i=0; i<vectorTam; i++){
        if(*x_max < x[i]){
            *x_max = x[i];
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Maximo elemento CPU: %lf\n", *x_max);
    printf("Tiempo CPU: %f ms\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    //liberar memoria
    free(x);
    free(x_max);
    cudaFree(d_x);
    cudaFree(d_x_max);
    cudaFree(d_buffer);
}