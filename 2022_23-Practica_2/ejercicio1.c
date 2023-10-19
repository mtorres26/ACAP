#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <ctype.h>
#include <string.h>

#define MASTER 0
#define DESTROY_TAG 666
#define NORMAL_TAG 1

void worker(int task, int rank){
    MPI_Status status;
    if(task==1)
    {
        if(rank == MASTER)
        {
            char* texto = malloc(sizeof(char)*20);
            printf("Introduzca un texto y pulse enter: \n");
            fflush(stdin);
            scanf("%s",texto);
            int tamcad = strlen(texto);
            MPI_Send(texto, tamcad, MPI_CHAR, 1, NORMAL_TAG, MPI_COMM_WORLD);
            MPI_Recv(texto, tamcad, MPI_CHAR, 1, NORMAL_TAG, MPI_COMM_WORLD, &status);         
            printf("El texto en mayuscula es: %s\n", texto);
            free(texto);
        }
        if(rank == 1)
        {
            int tamcad2;
            MPI_Probe(MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, MPI_CHAR, &tamcad2);
            char* cadena2 = malloc(sizeof(char)*tamcad2);
            MPI_Recv(cadena2, tamcad2, MPI_CHAR, MASTER, NORMAL_TAG, MPI_COMM_WORLD, &status);
            for(int i=0; i < tamcad2; i++)
            {
                cadena2[i] = toupper(cadena2[i]);
            }
            MPI_Send(cadena2, tamcad2, MPI_CHAR, MASTER, NORMAL_TAG, MPI_COMM_WORLD);
            free(cadena2);
        }
    }
    if(task==2)
    {
        if(rank==MASTER)
        {
            double* vector = malloc(sizeof(double)*10);
            for(int i = 1; i<11; i++){
                vector[i-1] = i+i*0.1;
            }
            MPI_Send(vector, 10, MPI_DOUBLE, 2, NORMAL_TAG, MPI_COMM_WORLD);
            MPI_Recv(vector, 2, MPI_DOUBLE, 2, NORMAL_TAG, MPI_COMM_WORLD, &status);
            fflush(stdout);
            printf("La suma total es %f y la raíz cuadrada es %f\n", vector[0], vector[1]);
            free(vector);
        }

        if(rank==2)
        {
            double* recibidos = malloc(sizeof(double)*10);
            double* resultados = malloc(sizeof(double)*2);
            MPI_Recv(recibidos, 10, MPI_DOUBLE, MASTER, NORMAL_TAG, MPI_COMM_WORLD, &status);
            for(int i = 0; i < 10; i++)
            {
                resultados[0] += recibidos[i];
            }
            resultados[1] = sqrt(resultados[0]);
            MPI_Send(resultados, 2, MPI_DOUBLE, MASTER, NORMAL_TAG, MPI_COMM_WORLD);
            free(recibidos);
            free(resultados);
        }
    }
    if(task==3)
    {
        if(rank==MASTER){
            int recvsuma = 0;
            MPI_Recv(&recvsuma, 1, MPI_INT, 3, NORMAL_TAG, MPI_COMM_WORLD, &status);
            fflush(stdout);
            printf("La suma de los enteros correspondientes a cada caracter del mensaje es: %d\n", recvsuma);
        }
        if(rank==3){
            char* mensaje = "Entrando en funcionalidad 3\n";
            int enterosmensaje[sizeof(mensaje)];
            int suma = 0;
            fflush(stdout); 
            printf("%s", mensaje);
            for(int i = 0; i < sizeof(mensaje); i++){
                enterosmensaje[i] = mensaje[i]; // Se asigna a cada entero el valor de cada caracter
                suma += enterosmensaje[i];
            }
            MPI_Send(&suma, 1, MPI_INT, MASTER, NORMAL_TAG, MPI_COMM_WORLD);
        }
    }
    if(task==4)
    {
        for(int i = 1; i < 4; i++){
            worker(i, rank); // Se llama a todas las task
        }
    }
}

int main(int argc, char* argv[])
{
    int rank, numProcs, task, tarea;
	MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;

    if(numProcs != 4){
        if(rank == MASTER)
        {
            printf("Error: 4 processes required\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }
	if(rank==MASTER){       
        do{
            printf("Introduzca numero de tarea: \n");
            fflush(stdin); // Vacio buffer de entrada
            scanf("%d", &task);
            if(task < 0 || task > 4){
                printf("Numero de tarea no válido. Rango válido: [0,4].\n");
            }else{
                MPI_Bcast(&task, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
                worker(task, rank);
            }
        }while(task != 0);
    }
    else{
        do{
            MPI_Bcast(&tarea, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
            worker(tarea, rank);
        }while(tarea != 0);
    }
    MPI_Finalize();
	return 0;
}