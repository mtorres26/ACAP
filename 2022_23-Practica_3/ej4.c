// Calculo aproximado de PI mediante la serie de Leibniz e integral del cuarto de circulo
// https://es.wikipedia.org/wiki/Serie_de_Leibniz
// N.C. Cruz, Universidad de Granada, 2023

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>
#include <mpi.h>

#define MASTER 0
#define DESTROY_TAG 666
#define NORMAL_TAG 1

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

typedef struct t_struct{
    int id;
    int repartoHilos;
	int numHilos;
} tareas;

double piRectangles(int intervals){
	double width = 1.0/intervals;
	double sum = 0.0, x;
	for(int i = 0; i<intervals; i++){
		x = (i + 0.5)*width;
		sum += 4.0/(1.0 + x*x);
	}
	return sum*width;
}

double* calculo = 0;
double* suma = 0;

void* body(void* param){
    tareas* laMia = (tareas*) param;

	double width = 1.0/laMia->repartoHilos;
	double x = 0.0, sum = 0.0;

	for(int i = laMia->id*laMia->repartoHilos; i<(laMia->id*laMia->repartoHilos)+laMia->repartoHilos; i++){
		x = (i + 0.5)*width;
		sum += 4.0/(1.0 + x*x);
	}

	pthread_mutex_lock(&lock);
	calculo[0] += sum*width;
	pthread_mutex_unlock(&lock);

	return calculo;
}

int main(int argc, char* argv[]){
    int rank, numProcs, intervalos, numThreads, repartoProcs;
	MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if(rank==MASTER){
		do{
			printf("Introduzca numero de intervalos: \n");
			scanf("%d", &intervalos);
		} while(intervalos < 1);
		do{
			printf("Introduzca numero de hilos por proceso (maximo 2): \n");
			scanf("%d", &numThreads);
		} while(numThreads < 1);

		repartoProcs = intervalos/numProcs;
		MPI_Bcast(&repartoProcs, 1, MPI_INT, MASTER, MPI_COMM_WORLD);


	} else {
		MPI_Bcast(&repartoProcs, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
	
	}

	pthread_t threads[numThreads]; 
	struct t_struct miTarea[numThreads];
	
	double aux0 = get_wall_time();

	calculo = malloc(sizeof(double));
	suma = malloc(sizeof(double));

	for(long int i = 0; i<numThreads; i++){
		miTarea[i].id = i;
		miTarea[i].repartoHilos = repartoProcs/numThreads; 
		miTarea[i].numHilos = numThreads;
		pthread_create(&threads[i], 0, body, &(miTarea[i]));
	}
		
	double calculoProcs;

	void* devuelto = 0;

	for(int i = 0; i < numThreads; i++){
		pthread_join(threads[i], &devuelto);
		calculoProcs = *((double*) devuelto);
	}

	if(rank==MASTER){
		double aux1 = get_wall_time();
		double tiempo = aux1-aux0;
		
		printf("PI por integración del círculo [%d intervalos] = \t%lf, tiempo =\t%lf\n", intervalos, calculoProcs, tiempo);
	} 

	free(devuelto);
	
    MPI_Finalize();

	return 0;
}