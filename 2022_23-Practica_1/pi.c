// Calculo aproximado de PI mediante la serie de Leibniz e integral del cuarto de circulo
// https://es.wikipedia.org/wiki/Serie_de_Leibniz
// N.C. Cruz, Universidad de Granada, 2023

#include <stdio.h>
#include <stdlib.h>

double piLeibniz(int steps){
	double partpi = 0.0;
	double num = 1.0;
	double denom = 1.0;
	for(int i = 0; i<steps; i++){
		partpi += num/denom;
		num = -1.0*num; // Alternamos el signo
		denom += 2.0;
	}
	return 4.0*partpi;
}

double piRectangles(int intervals){
	double width = 1.0/intervals;
	double sum = 0.0, x;
	for(int i = 0; i<intervals; i++){
		x = (i + 0.5)*width;
		sum += 4.0/(1.0 + x*x);
	}
	return sum*width;
}

int main(int argc, char* argv[]){
	if(argc!=2){	//El primer argumento siempre es el nombre del programa
		printf("Uso: ./prog esfuerzo\n");
	}else{
		int steps = atoi(argv[1]);
		if(steps<=0){
			printf("El número de iteraciones debe ser un entero positivo!\n");
		}else{
			double pi = piLeibniz(steps);
			printf("PI por la serie de G. Leibniz [%d iteraciones] =\t%lf\n", steps, pi);
			pi = piRectangles(steps);
			printf("PI por integración del círculo [%d intervalos] = \t%lf\n", steps, pi);
		}
	}
	return 0;
}
