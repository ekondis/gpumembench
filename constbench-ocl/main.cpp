/**
 * main.cpp: This file is part of the gpumembench micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <loclutil.h>
#include "const_kernels.h"

// Initialize vector data
void init_vector(int *v, size_t datasize){
	for(int i=0; i<(int)datasize; i++){
		v[i] = 1;
	}
}

int main(int argc, char* argv[]) {
	printf("OpenCL constbench (constant memory bandwidth micro-benchmark)\n");

	unsigned int datasize = VECTOR_SIZE*sizeof(int);

	cl_device_id clDev = loclGetArgSelectedDevice(argc, argv);
	if( clDev == (cl_device_id)-1 ){
		loclPrintAvailableDevices();
		exit(EXIT_FAILURE);
	}

	int *a = (int*)malloc(datasize);
	init_vector(a, VECTOR_SIZE);

	// benchmark execution
	constbenchGPU(clDev, a, 4096*VECTOR_SIZE);

	free(a);

	return 0;
}