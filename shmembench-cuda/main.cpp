/**
 * main.cpp: This file is part of the gpumembench micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>
#include <lcutil.h>
#include "shmem_kernels.h"

#define VECTOR_SIZE (1024*1024)

// Initialize vector data
void init_vector(double *v, size_t datasize){
	for(int i=0; i<(int)datasize; i++){
		v[i] = i;
	}
}

int main(int argc, char* argv[]) {
	printf("CUDA shmembench (shared memory bandwidth microbenchmark)\n");

	unsigned int datasize = VECTOR_SIZE*sizeof(double);

	cudaSetDevice(0); // set first device as default

	StoreDeviceInfo(stdout);

	size_t freeCUDAMem, totalCUDAMem;
	cudaMemGetInfo(&freeCUDAMem, &totalCUDAMem);
	printf("Total GPU memory %lu, free %lu\n", totalCUDAMem, freeCUDAMem);

	printf("Buffer sizes: 3x%dMB\n", datasize/(1024*1024));
	
	double *a, *b, *c;
	a = (double*)malloc(datasize);
	b = (double*)malloc(datasize);
	c = (double*)malloc(datasize);
	init_vector(a, VECTOR_SIZE);
	init_vector(b, VECTOR_SIZE);
	memset(c, 0, sizeof(int)*VECTOR_SIZE);

	// benchmark execution
	shmembenchGPU(a, b, c, VECTOR_SIZE);

	free(a);
	free(b);
	free(c);

	return 0;
}

