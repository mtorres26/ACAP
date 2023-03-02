#include <stdio.h>
#include <mpi.h>

const int bufferSize = sizeof(int) + 2*sizeof(double); //Es lo que enviaremos

int main(int argc, char* argv[]){
	int rank, size;
	int position = 0;// Foco sobre el buffer (Ambos van a empezar en 0)
	char buffer[bufferSize]; // char -> byte
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if(size != 2){
		if(rank==0){
			printf("Ejecuta este programa con 2 procesos!\n");
		}
	}else{
		int primerDato = 0;
		double segundoDato[2];
		if(rank==0){
			primerDato = 3;
			segundoDato[0] = 4.4;
			segundoDato[1] = 5.5;
			MPI_Pack(&primerDato, 1, MPI_INT, buffer, bufferSize, &position, MPI_COMM_WORLD);
			MPI_Pack(segundoDato, 2, MPI_DOUBLE, buffer, bufferSize, &position, MPI_COMM_WORLD);
			MPI_Send(buffer, bufferSize, MPI_PACKED, 1, 0, MPI_COMM_WORLD);
		}else{
			MPI_Recv(buffer, bufferSize, MPI_PACKED, 0, 0, MPI_COMM_WORLD, &status);
			MPI_Unpack(buffer, bufferSize, &position, &primerDato, 1, MPI_INT, MPI_COMM_WORLD);
			MPI_Unpack(buffer, bufferSize, &position, segundoDato, 2, MPI_DOUBLE, MPI_COMM_WORLD);
			printf("Soy el proceso [%d] y he recibido esto:\n", rank);
			printf("primerDato=%d; segundoDato={%.1lf, %.1lf}:\n", primerDato, segundoDato[0], segundoDato[1]);
		}
	}
	
	MPI_Finalize();
	return 0;
}
