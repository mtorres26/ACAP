/*
   Línea de compilación: gcc main.c pgm.c -o main

   Ejemplo del uso de:

   pgmread
   pgmwrite

   Las matrices que contienen a las imágenes son declaradas como uchar **

   % % % % % % % %

   Ejemplo 1:

   unsigned char **Original = (unsigned char **)pgmread("entrada.pgm", &Largo, &Alto);
   La imagen PGM se lee de la línea de comando (argv[1]). La función
   pgmread regresa tres valores:

   1. la imagen leída       (Original)
   2. el largo de la imagen (Largo)
   3. el alto de la imagen  (Alto)

   % % % % % % % %

   Ejemplo 2:

   pgmwrite(Salida, "salida.pgm", Largo, Alto);
   La imagen Salida es escrita al disco con el nombre de negativo.pgm, la
   imagen resulta en formato PGM. La imagen se escribe desde el inicio (0,0)
   hasta (Largo, Alto).
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pgm.h"
#include <mpi.h>
#include <time.h>
#include <sys/time.h>

#define MASTER 0
#define NORMAL_TAG 1
#define DESTROY_TAG 666

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

void convolucion(unsigned char** Original, int** nucleo, unsigned char** Salida, int Largo, int Alto) {
  int x, y;
  int suma;
  int k = 0;
  int i, j;
  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++)
      k = k + nucleo[i][j];

  for (x = 1; x < Largo-1; x++){
    for (y = 1; y < Alto-1; y++){
      suma = 0;
      for (i = 0; i < 3; i++){
        for (j = 0; j < 3; j++){
            suma = suma + Original[(x-1)+i][(y-1)+j] * nucleo[i][j];
        }
      }
      if(k==0)
        Salida[x][y] = suma;
      else
        Salida[x][y] = suma/k;
    }
  }
}

/* * * * *          * * * * *          * * * * *          * * * * */

int main(int argc, char *argv[]){
  int rank, numProcs;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Status status;

  if(numProcs < 1){
    printf("Error: ejecutalo con 1 o mas procesos\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  int Largo, Alto;
  int i, j;
  
  int** nucleo = (int**) GetMem2D(3, 3, sizeof(int));
  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++)
      nucleo[i][j] = -1;
  nucleo[1][1] = 1;

  if(rank==MASTER){
    unsigned char** Original = pgmread("lena_original.pgm", &Largo, &Alto); // pgmread ya reserva memoria para Original

    MPI_Bcast(&Largo, 1, MPI_INT, MASTER, MPI_COMM_WORLD); 
    MPI_Bcast(&Alto, 1, MPI_INT, MASTER, MPI_COMM_WORLD); 
    // Enviamos las dimensiones de la imagen

    for(int i = 1; i < numProcs; i++){
      MPI_Send(*Original + i*(Alto*(Largo/numProcs)), Alto*(Largo/numProcs), MPI_UNSIGNED_CHAR, i, NORMAL_TAG, MPI_COMM_WORLD);
    } // Envia a cada proceso una parte distinta de la imagen

    unsigned char** Salida = (unsigned char**)GetMem2D(Largo, Alto, sizeof(unsigned char)); // Reservamos Largo x Alto, donde juntaremos 
                                                                                            // lo procesado por todos los procesos
    double aux = get_wall_time();
    convolucion(Original, nucleo, Salida, Largo/numProcs, Alto); // Escribimos en Salida solo la parte que le toca a Master, que es Largo/numProcs
    double aux2 = get_wall_time(); 
    double tiempo = aux2 - aux;
    printf("Proceso 0 tarda: %lf segundos.\n", tiempo);

    for(int i = 1; i < numProcs; i++){
      MPI_Recv(*Salida + i*(Alto*(Largo/numProcs)), Alto*(Largo/numProcs), MPI_UNSIGNED_CHAR, i, NORMAL_TAG, MPI_COMM_WORLD, &status); 
      printf("Proceso 0 recibe del proceso %d.\n", i);
    } // Recibe de cada proceso y lo guarda en Salida a partir de lo que ha llegado del proceso anterior 

    pgmwrite(Salida, "lena_hecha.pgm", Largo, Alto);

    Free2D((void**) Original, Largo);
    Free2D((void**) Salida, Largo);
  }
  else{
    int Largo_hijos;
    int Alto_hijos;
    MPI_Bcast(&Largo_hijos, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&Alto_hijos, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    
    unsigned char** reparto_imagen = (unsigned char**)GetMem2D(Largo_hijos/numProcs, Alto_hijos, sizeof(unsigned char));

    MPI_Recv(*reparto_imagen, Alto_hijos*(Largo_hijos/numProcs), MPI_UNSIGNED_CHAR, MASTER, NORMAL_TAG, MPI_COMM_WORLD, &status);
    // Recibe la parte que le haya enviado el Master
    
    unsigned char** Salida_hijos = (unsigned char**)GetMem2D(Largo_hijos/numProcs, Alto_hijos, sizeof(unsigned char));
    
    double aux = get_wall_time();
    convolucion(reparto_imagen, nucleo, Salida_hijos, Largo_hijos/numProcs, Alto_hijos);
    double aux2 = get_wall_time();
    double tiempo = aux2 - aux;
    printf("Proceso %i tarda: %lf segundos.\n", rank, tiempo);

    MPI_Send(*Salida_hijos, Alto_hijos*(Largo_hijos/numProcs), MPI_UNSIGNED_CHAR,  MASTER, NORMAL_TAG, MPI_COMM_WORLD);
    // Devuelve al Master lo procesado

    Free2D((void**) reparto_imagen, Largo_hijos/numProcs);
    Free2D((void**) Salida_hijos, Largo_hijos/numProcs);
  }

  Free2D((void**) nucleo, 3);

  MPI_Finalize();

  return (0);
}