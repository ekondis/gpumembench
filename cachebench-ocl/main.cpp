/**
 * main.cpp: This file is part of the gpumembench micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <loclutil.h>
#include "cache_kernels.h"

// Initialize vector data
void init_vector(double *v, size_t datasize){
	for(int i=0; i<(int)datasize; i++){
		v[i] = i;
	}
}

int main(int argc, char* argv[]) {
	printf("OpenCL cachebench (repeated memory cached operations microbenchmark)\n");

	printf("Syntax: cachebench-ocl <device-index> [all/l2/tex]\n");

	unsigned int datasize = VECTOR_SIZE*sizeof(cl_int4);

	cl_device_id clDev = loclGetArgSelectedDevice(argc, argv);
	if( clDev == (cl_device_id)-1 ){
		loclPrintAvailableDevices();
		exit(EXIT_FAILURE);
	}
	printf("Buffer size: %dMB\n", datasize/(1024*1024));
	
	double *c = (double*)malloc(datasize);
	init_vector(c, VECTOR_SIZE);

	bench_type bench_type = btAllCache;
	if( argc>2 ){
		if( strcmp(argv[2], "l2")==0 )
			bench_type = btL2Cache;
		else if( strcmp(argv[2], "tex")==0 )
			bench_type = btTexture;
		else if( strcmp(argv[2], "all")!=0 ){
			fprintf(stderr, "Error: Use all, l2 or tex memory type specifier\n");
			exit(1);
		}
	}
	// benchmark execution
	cachebenchGPU(clDev, c, VECTOR_SIZE, bench_type, true);

	free(c);

	return 0;
}
