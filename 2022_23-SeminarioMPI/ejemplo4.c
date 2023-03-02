#include <stdio.h>
#include <mpi.h>

int main(int argc, char* argv[])
{
    int rank, size;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(size != 2)
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
            
        }    
    }
}