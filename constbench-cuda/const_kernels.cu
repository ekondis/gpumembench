/**
 * const_kernels.cu: This file is part of the gpumembench micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <stdio.h>
#include <lcutil.h>
#include "const_kernels.h"

template <class T>
__device__ T init_vector(int v){
#pragma error "ERROR: Unimplemented!"
}

template <>
__device__ int init_vector(int v){
	return v;
}

template <>
__device__ int2 init_vector(int v){
	return make_int2(v, v);
}

template <>
__device__ int4 init_vector(int v){
	return make_int4(v, v, v, v);
}

template <class T>
__device__ int reduce_vector(T v){
#pragma error "ERROR: Unimplemented!"
	return 0;
}

template <>
__device__ int reduce_vector(int v){
	return v;
}

template <>
__device__ int reduce_vector(int2 v){
	return v.x + v.y;
}

template <>
__device__ int reduce_vector(int4 v){
	return v.x + v.y + v.z + v.w;
}

template <class T>
__device__ void add_vector(T &target, const T &v){
#pragma error "ERROR: Unimplemented!"
}

__device__ void add_vector(int &target, const int &v){
	target += v;
}

__device__ void add_vector(int2 &target, const int2 &v){
	target.x += v.x;
	target.y += v.y;
}

__device__ void add_vector(int4 &target, const int4 &v){
	target.x += v.x;
	target.y += v.y;
	target.z += v.z;
	target.w += v.w;
}

__device__ __constant__ int constant_data[VECTOR_SIZE];

template <class T>
__global__ void benchmark_constant(int *output){
	T* constant_data_p = (T*)constant_data;
	T sum = init_vector<T>(0);
	// Force 4 wide strides in order to avoid automatic merging of multiple accesses to 128bit accesses
	for(int i=0; i<4; i++){
#pragma unroll 128
		for(int j=0; j<VECTOR_SIZE/(sizeof(T)/sizeof(int)); j+=4){
			add_vector(sum, constant_data_p[j+i]);
#ifdef CUDA6_5_WORKAROUND
			add_vector(sum, init_vector<T>(1));
#endif
		}
	}
#ifdef CUDA6_5_WORKAROUND
	add_vector(sum, init_vector<T>(-(int)(VECTOR_SIZE/(sizeof(T)/sizeof(int)))));
#endif
	int res = reduce_vector(sum);
	if( threadIdx.x==0 && blockIdx.x==0 )
		*output = res;
}

void initializeEvents(cudaEvent_t *start, cudaEvent_t *stop){
	CUDA_SAFE_CALL( cudaEventCreate(start) );
	CUDA_SAFE_CALL( cudaEventCreate(stop) );
	CUDA_SAFE_CALL( cudaEventRecord(*start, 0) );
}

float finalizeEvents(cudaEvent_t start, cudaEvent_t stop){
	CUDA_SAFE_CALL( cudaGetLastError() );
	CUDA_SAFE_CALL( cudaEventRecord(stop, 0) );
	CUDA_SAFE_CALL( cudaEventSynchronize(stop) );
	float kernel_time;
	CUDA_SAFE_CALL( cudaEventElapsedTime(&kernel_time, start, stop) );
	CUDA_SAFE_CALL( cudaEventDestroy(start) );
	CUDA_SAFE_CALL( cudaEventDestroy(stop) );
	return kernel_time;
}

extern "C" void constbenchGPU(int *a, long gridsize){
	const int BLOCK_SIZE = 256;
	const int TOTAL_BLOCKS = gridsize/(BLOCK_SIZE);
	int *cd, c;

	CUDA_SAFE_CALL( cudaMalloc((void**)&cd, sizeof(int)) );

	// Copy data to device memory
	CUDA_SAFE_CALL( cudaMemcpyToSymbol(constant_data, a, VECTOR_SIZE*sizeof(int), 0, cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemset(cd, 0, sizeof(int)) );  // initialize to zeros

	// Synchronize in order to wait for memory operations to finish
	CUDA_SAFE_CALL( cudaDeviceSynchronize() );

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(TOTAL_BLOCKS);
	cudaEvent_t start, stop;

	// warm up
	benchmark_constant<int><<< dimGrid, dimBlock >>>(cd);
	CUDA_SAFE_CALL( cudaGetLastError() );
	CUDA_SAFE_CALL( cudaDeviceSynchronize() );

	CUDA_SAFE_CALL( cudaMemset(cd, 0, sizeof(int)) );  // initialize to zeros

	initializeEvents(&start, &stop);
	benchmark_constant<int><<< dimGrid, dimBlock >>>(cd);
	float krn_time_constant_32b = finalizeEvents(start, stop);

	// Copy results back to host memory and validate result
	CUDA_SAFE_CALL( cudaMemcpy(&c, cd, sizeof(int), cudaMemcpyDeviceToHost) );
	if( c!=VECTOR_SIZE ){
		fprintf(stderr, "ERROR: c was not %d (%d)!\n", VECTOR_SIZE, c);
		exit(1);
	}

	CUDA_SAFE_CALL( cudaMemset(cd, 0, sizeof(int)) );  // initialize to zeros

	initializeEvents(&start, &stop);
	benchmark_constant<int2><<< dimGrid, dimBlock >>>(cd);
	float krn_time_constant_64b = finalizeEvents(start, stop);

	CUDA_SAFE_CALL( cudaMemcpy(&c, cd, sizeof(int), cudaMemcpyDeviceToHost) );
	if( c!=VECTOR_SIZE ){
		fprintf(stderr, "ERROR: c was not %d (%d)!\n", VECTOR_SIZE, c);
		exit(1);
	}

	CUDA_SAFE_CALL( cudaMemset(cd, 0, sizeof(int)) );  // initialize to zeros

	initializeEvents(&start, &stop);
	benchmark_constant<int4><<< dimGrid, dimBlock >>>(cd);
	float krn_time_constant_128b = finalizeEvents(start, stop);

	CUDA_SAFE_CALL( cudaMemcpy(&c, cd, sizeof(int), cudaMemcpyDeviceToHost) );
	if( c!=VECTOR_SIZE ){
		fprintf(stderr, "ERROR: c was not %d (%d)!\n", VECTOR_SIZE, c);
		exit(1);
	}

	CUDA_SAFE_CALL( cudaFree(cd) );

	printf("Kernel execution time\n");
	printf("\tbenchmark_constant  (32bit):%10.4f msecs\n", krn_time_constant_32b);
	printf("\tbenchmark_constant  (64bit):%10.4f msecs\n", krn_time_constant_64b);
	printf("\tbenchmark_constant (128bit):%10.4f msecs\n", krn_time_constant_128b);
	
	printf("Total operations executed\n");
	const long long operations_bytes  = (long long)((VECTOR_SIZE))*gridsize*sizeof(int);
	const long long operations_32bit  = (long long)((VECTOR_SIZE))*gridsize/(sizeof(int)/sizeof(int));
	const long long operations_64bit  = (long long)((VECTOR_SIZE))*gridsize/(sizeof(int2)/sizeof(int));
	const long long operations_128bit = (long long)((VECTOR_SIZE))*gridsize/(sizeof(int4)/sizeof(int));
	printf("\tconstant memory array size :%12lu bytes\n", (VECTOR_SIZE)*sizeof(int));
	printf("\tconstant memory traffic    :%12.0f MB\n", (double)operations_bytes/(1000.*1000.));
	printf("\tconstant memory operations :%12lld operations (32bit)\n", operations_32bit);
	printf("\tconstant memory operations :%12lld operations (64bit)\n", operations_64bit);
	printf("\tconstant memory operations :%12lld operations (128bit)\n", operations_128bit);

	printf("Memory throughput\n");
	printf("\tusing  32bit operations :%8.2f GB/sec (%6.2f billion accesses/sec)\n", ( (double)operations_bytes)/krn_time_constant_32b*1000./(double)(1000.*1000.*1000.),  ((double)operations_32bit)/ krn_time_constant_32b*1000./(double)(1000*1000*1000));
	printf("\tusing  64bit operations :%8.2f GB/sec (%6.2f billion accesses/sec)\n", ( (double)operations_bytes)/krn_time_constant_64b*1000./(double)(1000.*1000.*1000.),  ((double)operations_64bit)/ krn_time_constant_64b*1000./(double)(1000*1000*1000));
	printf("\tusing 128bit operations :%8.2f GB/sec (%6.2f billion accesses/sec)\n", ((double)operations_bytes)/krn_time_constant_128b*1000./(double)(1000.*1000.*1000.), ((double)operations_128bit)/krn_time_constant_128b*1000./(double)(1000*1000*1000));

	printf("Normalized per SM\n");
	cudaDeviceProp deviceProp = GetDeviceProperties();
	printf("\tConstant memory operations per clock (32bit) :%8.2f (per SM%6.2f)\n",((double)operations_32bit)/(deviceProp.clockRate*krn_time_constant_32b), ((double)operations_32bit)/(deviceProp.clockRate*krn_time_constant_32b)/deviceProp.multiProcessorCount);
	printf("\tConstant memory operations per clock (64bit) :%8.2f (per SM%6.2f)\n",((double)operations_64bit)/(deviceProp.clockRate*krn_time_constant_64b), ((double)operations_64bit)/(deviceProp.clockRate*krn_time_constant_64b)/deviceProp.multiProcessorCount);
	printf("\tConstant memory operations per clock (128bit):%8.2f (per SM%6.2f)\n",((double)operations_128bit)/(deviceProp.clockRate*krn_time_constant_128b), ((double)operations_128bit)/(deviceProp.clockRate*krn_time_constant_128b)/deviceProp.multiProcessorCount);

	printf("Compute overhead\n");
	printf("\tAddition operations per constant memory operation  (32bit): 1\n");
	printf("\tAddition operations per constant memory operation  (64bit): 2\n");
	printf("\tAddition operations per constant memory operation (128bit): 4\n");
	CUDA_SAFE_CALL( cudaDeviceReset() );
}

