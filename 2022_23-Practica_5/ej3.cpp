#include <stdio.h>
#include <ctime>
#include <curand.h>
#include <cuda_runtime.h>

bool comparacionMatrices(double* m_a, double** m_b, int tam){
    bool iguales = true;
    for(int i = 0; i < tam; i++){
        for(int j = 0; j < tam; j++){
            if(round(m_a[i*tam+j]) != round(m_b[i][j])){
                iguales = false;
                printf("Valores distintos en [%d][%d]\n", i, j);
            }
        }
    }
    return iguales;
}

__global__ void producto_matrices(double* m_a, double* m_b, double* m_res, int tam){
    int fila = blockIdx.x; // Cada fila de m_res corresponden a un bloque
    int col = threadIdx.x; 
    int globalid = threadIdx.x + blockIdx.x * blockDim.x;
    double temp = 0.0;
    
    for(int i = 0; i < tam; i++){
        temp += m_a[fila*tam+i] * m_b[col+tam*i];
    }
    m_res[globalid] = temp;
}

int main(int argc, char* argv[]){
    if(argc != 2 || atoi(argv[1]) < 2){
        printf("Error:\n\t-Se necesita introducir un solo valor para el tamaño de las matrices.\n");
        exit(1);
    }
    int tam = atoi(argv[1]);
    double *m_a, *m_b, *d_m_a, *d_m_b;
    double *m_res, *d_m_res;
    m_a = (double*)malloc(tam*tam*sizeof(double));
    m_b = (double*)malloc(tam*tam*sizeof(double));
    m_res = (double*)malloc(tam*tam*sizeof(double));
    cudaMalloc((void**)&d_m_a, tam*tam*sizeof(double));
    cudaMalloc((void**)&d_m_b, tam*tam*sizeof(double));
    cudaMalloc((void**)&d_m_res, tam*tam*sizeof(double));
    cudaMemset(d_m_res, 0, tam*tam*sizeof(double));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    for(int i = 0; i < tam; i++){
        for(int j = 0; j < tam; j++){
            m_a[i*tam+j] = rand()/1e5;
            m_b[i*tam+j] = rand()/1e5;
            m_res[i*tam+j] = 0;
        }
    }

    cudaEventRecord(start);

    cudaMemcpy(d_m_a, m_a, tam*tam*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m_b, m_b, tam*tam*sizeof(double), cudaMemcpyHostToDevice);

    int blockTam = tam;
    int gridTam = tam;

    producto_matrices<<<gridTam, blockTam>>>(d_m_a, d_m_b, d_m_res, tam);

    cudaDeviceSynchronize();
    cudaMemcpy(m_res, d_m_res, tam*tam*sizeof(double), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float millisecondsGPU = 0;
    cudaEventElapsedTime(&millisecondsGPU, start, stop);    

    // Comprobacion secuencial
    double **m_cpu;
    m_cpu = (double**)malloc (tam*sizeof(double*));
    for (int i=0;i<tam;i++){
        m_cpu[i] = (double*)malloc(tam*sizeof(double));
    }
    cudaEventRecord(start);

    for(int i = 0; i < tam; i++){
        for(int j = 0; j < tam; j++){
            m_cpu[i][j] = 0;
        }
    }

    for(int i = 0; i < tam; i++){
        for(int j = 0; j < tam; j++){
            for(int k = 0; k < tam; k++){
                m_cpu[i][j] += m_a[i*tam+k]*m_b[k*tam+j];
                //printf("m_cpu[%d] += m_a[%d] * m_b[%d]\n", i*tam+j, i*tam+k, k*tam+j);
            }
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float millisecondsCPU = 0;
    cudaEventElapsedTime(&millisecondsCPU, start, stop);

    if(tam < 7){
        printf("Resultado producto GPU:\n");
        for(int i = 0; i < tam; i++){
            for(int j = 0; j < tam; j++){
                printf("%lf ", m_res[i*tam+j]);
            }
            printf("\n");
            fflush(stdout);
        }
        printf("\n");
        printf("Resultado producto CPU:\n");
        for(int i = 0; i < tam; i++){
            for(int j = 0; j < tam; j++){
                printf("%lf ", m_cpu[i][j]);
            }
            printf("\n");
            fflush(stdout);
        }
    } 
    if(comparacionMatrices(m_res, m_cpu, tam)){
        printf("Los resultados son practicamente iguales.\n");
    } else {
        printf("Los resultados NO son iguales.\n");
    }
    printf("Tiempo GPU: %lf\n", millisecondsGPU);
    printf("Tiempo CPU: %lf\n", millisecondsCPU);
    

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(m_a);
    free(m_b);
    free(m_res);
    cudaFree(d_m_a);
    cudaFree(d_m_b);
    cudaFree(d_m_res);
}
