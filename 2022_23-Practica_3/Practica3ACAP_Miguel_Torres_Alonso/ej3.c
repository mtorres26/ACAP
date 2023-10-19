#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>

const int NUM_THREADS = 4;
pthread_mutex_t lock;
pthread_barrier_t barrera;

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

int* vectorA = 0;
int* vectorB = 0;
int* intersecAB = 0;
int* unionAB = 0;
double* jaccard = 0;

typedef struct t_struct{
    int id;
    int numHilos;
    int longitA;
    int longitB;
    int reparto;
    int resto;
    int cont;
} tareas;

void* body(void* param){
    tareas* laMia = (tareas*) param;
    
    // Funcionamiento explicado en documentación 
    for(int i = 0; i < laMia->reparto; i++){
        laMia->cont = 0;
        pthread_mutex_lock(&lock);
        unionAB[0]+=1;
        pthread_mutex_unlock(&lock);
        for(int j = 0; j < laMia->longitB; j++){
            if(vectorA[i + laMia->id*laMia->reparto] == vectorB[j]){
                pthread_mutex_lock(&lock);
                intersecAB[0] += 1;
                pthread_mutex_unlock(&lock);
            } else if(laMia->cont != laMia->longitB){
                laMia->cont++;
            }
            if(laMia->cont == laMia->longitB){
                pthread_mutex_lock(&lock);
                unionAB[0]+=1;
                pthread_mutex_unlock(&lock);
            }
        }
    }
    if(laMia->resto != 0 && laMia->id == laMia->numHilos -1){
        for(int i = laMia->longitA - laMia->resto; i < laMia->longitA; i++){
            laMia->cont = 0;
            pthread_mutex_lock(&lock);
            unionAB[0]+=1;
            pthread_mutex_unlock(&lock);
            for(int j = 0; j < laMia->longitB; j++){
                if(vectorA[i] == vectorB[j]){
                    pthread_mutex_lock(&lock);
                    intersecAB[0]+=1;
                    pthread_mutex_unlock(&lock);
                } else if(laMia->cont != laMia->longitB){
                    laMia->cont++;
                }
                if(laMia->cont == laMia->longitB){
                    pthread_mutex_lock(&lock);
                    unionAB[0]+=1;
                    pthread_mutex_unlock(&lock);
                }
            }
        }
    }
    // Barrera para que todos calculen Jaccard con los mismos valores
    pthread_barrier_wait(&barrera);
    
    jaccard[0] = (double) (intersecAB[0]) / (double) (unionAB[0]);

    return jaccard;
}

int main(int argc, char* argv[]){
    
    int longitudA, longitudB;
    int numThreads;
    do{
        fflush(stdout);
        printf("Introduzca longitud del vector A de valores (no negativa):\n");
        fflush(stdin);
        scanf("%d", &longitudA);
    } while(longitudA < 1);
    do{
        fflush(stdout);
        printf("Introduzca longitud del vector B de valores (no negativa):\n");
        fflush(stdin);
        scanf("%d", &longitudB);
    } while(longitudB < 1);

    do{
        fflush(stdout);
        printf("Introduzca numero de hilos a desplegar (no negativo):\n");
        fflush(stdin);
        scanf("%d", &numThreads);
    } while(numThreads < 1);

    // Reservas de memoria
    vectorA = malloc(sizeof(int)*longitudA);
    vectorB = malloc(sizeof(int)*longitudB);
    unionAB = malloc(sizeof(int));
    intersecAB = malloc(sizeof(int));
    jaccard = malloc(sizeof(double));

    // Rellenar vectores
    for(int i = 0; i < longitudA; i++){
        vectorA[i] = i;
    }
    for(int i = 0; i < longitudB; i++){
        vectorB[i] = 2*i;
    }
    
    pthread_t threads[numThreads];
    struct t_struct miTarea[numThreads];
    pthread_barrier_init(&barrera,0, numThreads);

    double aux1 = get_wall_time();
    for(long int i = 0; i<numThreads; i++){
        miTarea[i].id = i;
        miTarea[i].numHilos = numThreads;
        miTarea[i].longitA = longitudA;
        miTarea[i].longitB = longitudB;
        miTarea[i].reparto = longitudA/numThreads;
        miTarea[i].resto = longitudA % numThreads;
        pthread_create(&threads[i], 0, body, &(miTarea[i]));
    }

    void* devuelto = 0;

    for(int i = 0; i < numThreads; i++){
        pthread_join(threads[i], &devuelto);
    }
    double aux2 = get_wall_time();
    double tiempo = aux2 - aux1;
    printf("La distancia de Jaccard calculada es %lf y se han empleado %lf segundos.\n", *((double*) devuelto), tiempo);

    pthread_barrier_destroy(&barrera);
    free(devuelto);
    free(vectorA);
    free(vectorB);
    free(unionAB);
    free(intersecAB);
    //free(jaccard);

    pthread_exit(0); 

}