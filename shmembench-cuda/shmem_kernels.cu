/**
 * shmem_kernels.cu: This file is part of the gpumembench micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <stdio.h>
#include <lcutil.h>

#define TOTAL_ITERATIONS (1024)

// shared memory swap operation (2 floats read + 2 floats write)
template <class T>
__device__ void shmem_swap(T *v1, T *v2){
	T tmp;
	tmp = *v2;
	*v2 = *v1;
	*v1 = tmp;
}

template <class T>
__device__ T init_val(int i){
#pragma error "kop"
}

template <>
__device__ float init_val(int i){
	return i;
}

template <>
__device__ float2 init_val(int i){
	return make_float2(i, i+11);
}

template <>
__device__ float4 init_val(int i){
	return make_float4(i, i+11, i+19, i+23);
}

template <class T>
__device__ T reduce_vector(T v1, T v2, T v3, T v4, T v5, T v6){
#pragma error "rrrrrr!"
}

template <>
__device__ float reduce_vector(float v1, float v2, float v3, float v4, float v5, float v6){
	return v1 + v2 + v3 + v4 + v5 + v6;
}

template <>
__device__ float2 reduce_vector(float2 v1, float2 v2, float2 v3, float2 v4, float2 v5, float2 v6){
	return make_float2(v1.x + v2.x + v3.x + v4.x + v5.x + v6.x, v1.y + v2.y + v3.y + v4.y + v5.y + v6.y);
}

template <>
__device__ float4 reduce_vector(float4 v1, float4 v2, float4 v3, float4 v4, float4 v5, float4 v6){
	return make_float4(v1.x + v2.x + v3.x + v4.x + v5.x + v6.x, v1.y + v2.y + v3.y + v4.y + v5.y + v6.y, v1.z + v2.z + v3.z + v4.z + v5.z + v6.z, v1.w + v2.w + v3.w + v4.w + v5.w + v6.w);
}

template <class T>
__device__ void set_vector(T *target, int offset, T v){
#pragma error "Unimplemented!"
}

__device__ void set_vector(float *target, int offset, float v){
	target[offset] = v;
}

__device__ void set_vector(float2 *target, int offset, float2 v){
	target[offset].x = v.x;
	target[offset].y = v.y;
}

__device__ void set_vector(float4 *target, int offset, float4 v){
	target[offset].x = v.x;
	target[offset].y = v.y;
	target[offset].z = v.z;
	target[offset].w = v.w;
}

extern __shared__ float shm_buffer_ptr[];

template <class T>
__global__ void benchmark_shmem(T *g_data){
	T *shm_buffer = (T*)shm_buffer_ptr;
	int tid = threadIdx.x; 
	int globaltid = blockIdx.x*blockDim.x + tid;
	set_vector(shm_buffer, tid+0*blockDim.x, init_val<T>(tid));
	set_vector(shm_buffer, tid+1*blockDim.x, init_val<T>(tid+1));
	set_vector(shm_buffer, tid+2*blockDim.x, init_val<T>(tid+3));
	set_vector(shm_buffer, tid+3*blockDim.x, init_val<T>(tid+7));
	set_vector(shm_buffer, tid+4*blockDim.x, init_val<T>(tid+13));
	set_vector(shm_buffer, tid+5*blockDim.x, init_val<T>(tid+17));
	__threadfence_block();
#pragma unroll 32
	for(int j=0; j<TOTAL_ITERATIONS; j++){
		shmem_swap(shm_buffer+tid+0*blockDim.x, shm_buffer+tid+1*blockDim.x);
		shmem_swap(shm_buffer+tid+2*blockDim.x, shm_buffer+tid+3*blockDim.x);
		shmem_swap(shm_buffer+tid+4*blockDim.x, shm_buffer+tid+5*blockDim.x);
		__threadfence_block();
		shmem_swap(shm_buffer+tid+1*blockDim.x, shm_buffer+tid+2*blockDim.x);
		shmem_swap(shm_buffer+tid+3*blockDim.x, shm_buffer+tid+4*blockDim.x);
		__threadfence_block();
	}
	g_data[globaltid] = reduce_vector<T>(shm_buffer[tid+0*blockDim.x], shm_buffer[tid+1*blockDim.x], shm_buffer[tid+2*blockDim.x], shm_buffer[tid+3*blockDim.x], shm_buffer[tid+4*blockDim.x], shm_buffer[tid+5*blockDim.x]);
}

double max3(double v1, double v2, double v3){
	double t = v1>v2 ? v1 : v2;
	return t>v3 ? t : v3;
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

extern "C" void shmembenchGPU(double *c, long size){
	const int BLOCK_SIZE = 256;
	const int TOTAL_BLOCKS = size/(BLOCK_SIZE);
	double *cd;

	CUDA_SAFE_CALL( cudaMalloc((void**)&cd, size*sizeof(double)) );

	// Copy data to device memory
	CUDA_SAFE_CALL( cudaMemset(cd, 0, size*sizeof(double)) );  // initialize to zeros

	// Synchronize in order to wait for memory operations to finish
	CUDA_SAFE_CALL( cudaDeviceSynchronize() );

	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid_f1(TOTAL_BLOCKS, 1, 1);
	dim3 dimGrid_f2(TOTAL_BLOCKS/2, 1, 1);
	dim3 dimGrid_f4(TOTAL_BLOCKS/4, 1, 1);
	int shared_mem_per_block = BLOCK_SIZE*sizeof(float)*6;
	cudaEvent_t start, stop;

	// warm up
	benchmark_shmem<float><<< dimGrid_f4, dimBlock, shared_mem_per_block >>>((float*)cd);
	CUDA_SAFE_CALL( cudaGetLastError() );
	CUDA_SAFE_CALL( cudaDeviceSynchronize() );

	initializeEvents(&start, &stop);
	benchmark_shmem<float><<< dimGrid_f1, dimBlock, shared_mem_per_block >>>((float*)cd);
	float krn_time_shmem_32b = finalizeEvents(start, stop);

	initializeEvents(&start, &stop);
	benchmark_shmem<float2><<< dimGrid_f2, dimBlock, shared_mem_per_block*2 >>>((float2*)cd);
	float krn_time_shmem_64b = finalizeEvents(start, stop);

	initializeEvents(&start, &stop);
	benchmark_shmem<float4><<< dimGrid_f4, dimBlock, shared_mem_per_block*4 >>>((float4*)cd);
	float krn_time_shmem_128b = finalizeEvents(start, stop);

	// Copy results back to host memory
	CUDA_SAFE_CALL( cudaMemcpy(c, cd, size*sizeof(double), cudaMemcpyDeviceToHost) );

	CUDA_SAFE_CALL( cudaFree(cd) );

	printf("Kernel execution time\n");
	printf("\tbenchmark_shmem  (32bit):%10.3f msecs\n", krn_time_shmem_32b);
	printf("\tbenchmark_shmem  (64bit):%10.3f msecs\n", krn_time_shmem_64b);
	printf("\tbenchmark_shmem (128bit):%10.3f msecs\n", krn_time_shmem_128b);
	
	printf("Total operations executed\n");
	const long long operations_bytes  = (6LL+4*5*TOTAL_ITERATIONS+6)*size*sizeof(float);
	const long long operations_32bit  = (6LL+4*5*TOTAL_ITERATIONS+6)*size;
	const long long operations_64bit  = (6LL+4*5*TOTAL_ITERATIONS+6)*size/2;
	const long long operations_128bit = (6LL+4*5*TOTAL_ITERATIONS+6)*size/4;
	printf("\tshared memory traffic    :%12.0f GB\n", (double)operations_bytes/(1000.*1000.*1000.));
	printf("\tshared memory operations :%12lld operations (32bit)\n", operations_32bit);
	printf("\tshared memory operations :%12lld operations (64bit)\n", operations_64bit);
	printf("\tshared memory operations :%12lld operations (128bit)\n", operations_128bit);

	printf("Memory throughput\n");
	printf("\tusing  32bit operations   :%8.2f GB/sec (%6.2f billion accesses/sec)\n", ( (double)operations_bytes)/krn_time_shmem_32b*1000./(double)(1000.*1000.*1000.),  ((double)operations_32bit)/ krn_time_shmem_32b*1000./(double)(1000*1000*1000));
	printf("\tusing  64bit operations   :%8.2f GB/sec (%6.2f billion accesses/sec)\n", ( (double)operations_bytes)/krn_time_shmem_64b*1000./(double)(1000.*1000.*1000.),  ((double)operations_64bit)/ krn_time_shmem_64b*1000./(double)(1000*1000*1000));
	printf("\tusing 128bit operations   :%8.2f GB/sec (%6.2f billion accesses/sec)\n", ((double)operations_bytes)/krn_time_shmem_128b*1000./(double)(1000.*1000.*1000.), ((double)operations_128bit)/krn_time_shmem_128b*1000./(double)(1000*1000*1000));
	printf("\tpeak operation throughput :%8.2f Giga ops/sec\n",
		max3(((double)operations_32bit)/ krn_time_shmem_32b*1000./(double)(1000*1000*1000),
			((double)operations_64bit)/ krn_time_shmem_64b*1000./(double)(1000*1000*1000),
			((double)operations_128bit)/krn_time_shmem_128b*1000./(double)(1000*1000*1000)));

	printf("Normalized per SM\n");
	cudaDeviceProp deviceProp = GetDeviceProperties();
	printf("\tshared memory operations per clock (32bit) :%8.2f (per SM%6.2f)\n",((double)operations_32bit)/(deviceProp.clockRate*krn_time_shmem_32b), ((double)operations_32bit)/(deviceProp.clockRate*krn_time_shmem_32b)/deviceProp.multiProcessorCount);
	printf("\tshared memory operations per clock (64bit) :%8.2f (per SM%6.2f)\n",((double)operations_64bit)/(deviceProp.clockRate*krn_time_shmem_64b), ((double)operations_64bit)/(deviceProp.clockRate*krn_time_shmem_64b)/deviceProp.multiProcessorCount);
	printf("\tshared memory operations per clock (128bit):%8.2f (per SM%6.2f)\n",((double)operations_128bit)/(deviceProp.clockRate*krn_time_shmem_128b), ((double)operations_128bit)/(deviceProp.clockRate*krn_time_shmem_128b)/deviceProp.multiProcessorCount);
	CUDA_SAFE_CALL( cudaDeviceReset() );
}
