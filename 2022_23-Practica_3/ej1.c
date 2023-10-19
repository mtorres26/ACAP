#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

const int NUM_THREADS = 4;

typedef struct t_struct{
    int id;
    int longit;
    int reparto;
    double* vectorval;
    int resto;
} tareas;

void* body(void* param){
    tareas* laMia = (tareas*) param;
    double* maximo = malloc(sizeof(double));
    
    for(int i = 0; i < laMia->reparto; i++){
        if(laMia->vectorval[i + laMia->id*laMia->reparto] > maximo[0]){
            maximo[0] = laMia->vectorval[i + laMia->id*laMia->reparto];
        }
    }

    if(laMia->resto != 0 && laMia->id == NUM_THREADS -1){
        for(int i = laMia->longit - laMia->resto; i < laMia->longit; i++){
            if(laMia->vectorval[i] > maximo[0]){
                maximo[0] = laMia->vectorval[i];
            }
        }
    }
    return &(maximo[0]);
}

int main(int argc, char* argv[]){
    
    int longitud;
    do{
        fflush(stdout);
        printf("Introduzca longitud del vector de valores (no negativa):\n");
        fflush(stdin);
        scanf("%d", &longitud);
    } while(longitud < 1);
    

    double *vector = malloc(sizeof(double)*longitud);
    
    srand(time(NULL));
    for(int i = 0; i < longitud; i++){ 
        // Se rellena el vector
        vector[i] = rand();
    }

    pthread_t threads[NUM_THREADS];
    struct t_struct miTarea[NUM_THREADS];
    for(long int i = 0; i<NUM_THREADS; i++){
        miTarea[i].id = i;
        miTarea[i].longit = longitud;
        miTarea[i].reparto = longitud/NUM_THREADS;
        miTarea[i].vectorval = vector;
        miTarea[i].resto = longitud % NUM_THREADS;
        pthread_create(&threads[i], 0, body, &(miTarea[i]));
    }

    double* max = malloc(sizeof(double));

    void* devuelto = malloc(sizeof(double));

    for(int i = 0; i < NUM_THREADS; i++){
        pthread_join(threads[i], &devuelto);
        if(*((double*) devuelto) > max[0]){
            max[0] = *((double*) devuelto);
        }
    }
    printf("El maximo buscado con hilos es: %lf\n", max[0]);


    // Busqueda secuencial


    max[0] = 0;
    for(int i = 0; i < longitud; i++){
        if(vector[i] > max[0]){
            max[0] = vector[i];
        }
    }
    fflush(stdout);
    printf("Máximo buscado secuencialmente es: %lf\n", max[0]);
    
    free(vector);
    free(devuelto);
    free(max);

    pthread_exit(0); 
}