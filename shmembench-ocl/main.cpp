/**
 * main.cpp: This file is part of the gpumembench micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <loclutil.h>
#include "shmem_kernels.h"

#define VECTOR_SIZE (1024*1024)

int main(int argc, char* argv[]) {
	printf("OpenCL shmembench (local memory bandwidth micro-benchmark)\n");

	unsigned int datasize = VECTOR_SIZE*sizeof(int);

	cl_device_id clDev = loclGetArgSelectedDevice(argc, argv);
	if( clDev == (cl_device_id)-1 ){
		loclPrintAvailableDevices();
		exit(EXIT_FAILURE);
	}

	printf("Buffer sizes: %dMB\n", datasize/(1024*1024));
	
	int *c;
	c = (int*)malloc(datasize);
	memset(c, 0, sizeof(int)*VECTOR_SIZE);

	// benchmark execution
	shmembenchGPU(clDev, c, VECTOR_SIZE);

	free(c);
	return 0;
}

