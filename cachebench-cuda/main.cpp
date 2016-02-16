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
#include "cache_kernels.h"

#define VECTOR_SIZE (/*4*/32*1024*1024)

// Initialize vector data
void init_vector(double *v, size_t datasize){
	for(int i=0; i<(int)datasize; i++){
		v[i] = i;
	}
}


int main(int argc, char* argv[]) {
	printf("CUDA cachebench (repeated memory cached operations microbenchmark)\n");

	unsigned int datasize = VECTOR_SIZE*sizeof(int4); // reserve space for int4 types

	cudaSetDevice(0); // set first device as default

	StoreDeviceInfo(stdout);

	size_t freeCUDAMem, totalCUDAMem;
	cudaMemGetInfo(&freeCUDAMem, &totalCUDAMem);
	printf("Total GPU memory %lu, free %lu\n", totalCUDAMem, freeCUDAMem);

	printf("Buffer size: %dMB\n", datasize/(1024*1024));
	
	double *c = (double*)malloc(datasize);
	init_vector(c, VECTOR_SIZE);

	// benchmark execution
	cachebenchGPU(c, VECTOR_SIZE, true);

	free(c);

	return 0;
}

