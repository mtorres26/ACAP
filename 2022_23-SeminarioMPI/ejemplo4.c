#include <stdio.h>
#include <mpi.h>

const int tam = 10;

int main(int argc, char* argv[])
{
    int rank, size, v[tam];
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(size < 2)
    {
        if(rank==0)
        {
            "Ejecuta este programa con 2 procesos. \n";       
        }
    }
    else
    {
        if(rank == 0)
        {
            for(int i = 0; i < tam-1; i++)
            {
                v[i] = i;
                MPI_Send(v, tam, MPI_INT, i%size, 0, MPI_COMM_WORLD);
            }
        }
        if(rank != MASTER)
        {
            
        } 
    }


    MPI_Finalize();
}