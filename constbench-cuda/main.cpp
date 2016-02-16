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
#include "const_kernels.h"

// Initialize vector data
void init_vector(int *v, size_t datasize){
	for(int i=0; i<(int)datasize; i++){
		v[i] = 1;
	}
}

int main(int argc, char* argv[]) {
	printf("constbench (constant memory bandwidth microbenchmark)\n");

	unsigned int datasize = VECTOR_SIZE*sizeof(int);

	cudaSetDevice(0); // set first device as default

	StoreDeviceInfo(stdout);

	size_t freeCUDAMem, totalCUDAMem;
	cudaMemGetInfo(&freeCUDAMem, &totalCUDAMem);
	printf("Total GPU memory %lu, free %lu\n", totalCUDAMem, freeCUDAMem);

	int *a = (int*)malloc(datasize);
	init_vector(a, VECTOR_SIZE);

	// benchmark execution
	constbenchGPU(a, 4096*VECTOR_SIZE);

	free(a);

	return 0;
}

