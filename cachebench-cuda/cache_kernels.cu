/**
 * cache_kernels.cu: This file is part of the gpumembench micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <stdio.h>
#include <lcutil.h>

#define TOTAL_ITERATIONS  (8192)
#define UNROLL_ITERATIONS (64)

#define UNROLL_ITERATIONS_MEM (UNROLL_ITERATIONS/2)

const int BLOCK_SIZE = 256;

texture< int, 1, cudaReadModeElementType> texdataI1;
texture<int2, 1, cudaReadModeElementType> texdataI2;
texture<int4, 1, cudaReadModeElementType> texdataI4;

template<class T>
class dev_fun{
public:
	// Pointer displacement operation
	__device__ unsigned int operator()(T v1, unsigned int v2);
	// Compute operation (#1)
	__device__ T operator()(const T &v1, const T &v2);
	// Compute operation (#2)
	__device__ T comp_per_element(const T &v1, const T &v2);
	// Value initialization
	__device__ T init(int v);
	// Element loading
	__device__ T load(volatile const T* p, unsigned int offset);
	// Element storing
	__device__ void store(volatile T* p, unsigned int offset, const T &value);
	// Get first element
	__device__ int first_element(const T &v);
	// Reduce elements (XOR operation)
	__device__ int reduce(const T &v);
};


template<>
__device__ unsigned int dev_fun<int>::operator()(int v1, unsigned int v2){
	return v2+(unsigned int)v1 ;
}
template<>
__device__ int dev_fun<int>::operator()(const int &v1, const int &v2){
  return v1 + v2;
}
template<>
__device__ int dev_fun<int>::comp_per_element(const int &v1, const int &v2){
  return v1 - v2;
}
template<>
__device__ int dev_fun<int>::init(int v){
	return v;
}
template<>
__device__ int dev_fun<int>::load(volatile const int* p, unsigned int offset){
	int retval;
#ifdef TEX_LOADS
	retval = tex1Dfetch(texdataI1, offset);
#else
	p += offset;
	// Cache Operators for Memory Load Instructions
	// .ca Cache at all levels, likely to be accessed again.
	// .cg Cache at global level (cache in L2 and below, not L1).
	// .cs Cache streaming, likely to be accessed once.
	// .cv Cache as volatile (consider cached system memory lines stale, fetch again).
#ifdef L2_ONLY
	// Global level caching
	asm volatile ("ld.cg.u32 %0, [%1];" : "=r"(retval) : "l"(p));
#else
	// All cache levels utilized
	asm volatile ("ld.ca.u32 %0, [%1];" : "=r"(retval) : "l"(p));
#endif
#endif
	return retval;
}
template<>
__device__ void dev_fun<int>::store(volatile int* p, unsigned int offset, const int &value){
	p += offset;
	// Cache Operators for Memory Store Instructions
	// .wb Cache write-back all coherent levels.
	// .cg Cache at global level (cache in L2 and below, not L1).
	// .cs Cache streaming, likely to be accessed once.
	// .wt Cache write-through (to system memory).

	// Streaming store
	asm volatile ("st.cs.global.u32 [%0], %1;" :: "l"(p), "r"(value));
}
template<>
__device__ int dev_fun<int>::first_element(const int &v){
	return v;
}
template<>
__device__ int dev_fun<int>::reduce(const int &v){
	return v;
}


template<>
__device__ unsigned int dev_fun<int2>::operator()(int2 v1, unsigned int v2){
	return v2+(unsigned int)(v1.x+v1.y) ;
}
template<>
__device__ int2 dev_fun<int2>::operator()(const int2 &v1, const int2 &v2){
	return make_int2(v1.x + v2.x, v1.y + v2.y);
}
template<>
__device__ int2 dev_fun<int2>::comp_per_element(const int2 &v1, const int2 &v2){
	return make_int2(v1.x - v2.x, v1.y - v2.y);
}
template<>
__device__ int2 dev_fun<int2>::init(int v){
	return make_int2(v, v);
}
template<>
__device__ int2 dev_fun<int2>::load(volatile const int2* p, unsigned int offset){
	union{
		unsigned long long ll;
		int2 i2;
	} retval;
#ifdef TEX_LOADS
	retval.i2 = tex1Dfetch(texdataI2, offset);
#else
	p += offset;
#ifdef L2_ONLY
	// Global level caching
	asm volatile ("ld.cg.u64 %0, [%1];" : "=l"(retval.ll) : "l"(p));
#else
	// All cache levels utilized
	asm volatile ("ld.ca.u64 %0, [%1];" : "=l"(retval.ll) : "l"(p));
#endif
#endif
	return retval.i2;
}
template<>
__device__ void dev_fun<int2>::store(volatile int2* p, unsigned int offset, const int2 &value){
	union{
		unsigned long long ll;
		int2 i2;
	} retval;
	retval.i2 = value;
	p += offset;
	// Streaming store
	asm volatile ("st.cs.global.u64 [%0], %1;" :: "l"(p), "l"(retval.ll));
}
template<>
__device__ int dev_fun<int2>::first_element(const int2 &v){
	return v.x;
}
template<>
__device__ int dev_fun<int2>::reduce(const int2 &v){
	return v.x ^ v.y;
}


template<>
__device__ unsigned int dev_fun<int4>::operator()(int4 v1, unsigned int v2){
	return v2+(unsigned int)(v1.x+v1.y+v1.z+v1.w) ;
}
template<>
__device__ int4 dev_fun<int4>::operator()(const int4 &v1, const int4 &v2){
	return make_int4(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w);
}
template<>
__device__ int4 dev_fun<int4>::comp_per_element(const int4 &v1, const int4 &v2){
	return make_int4(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.w - v2.w);
}
template<>
__device__ int4 dev_fun<int4>::init(int v){
	return make_int4(v, v, v, v);
}
template<>
__device__ int4 dev_fun<int4>::load(volatile const int4* p, unsigned int offset){
	int4 retval;
#ifdef TEX_LOADS
	retval = tex1Dfetch(texdataI4, offset);
#else
	p += offset;
#ifdef L2_ONLY
	// Global level caching
	asm volatile ("ld.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(retval.x), "=r"(retval.y), "=r"(retval.z), "=r"(retval.w) : "l"(p));
#else
	// All cache levels utilized
	asm volatile ("ld.ca.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(retval.x), "=r"(retval.y), "=r"(retval.z), "=r"(retval.w) : "l"(p));
#endif
#endif
	return retval;
}
template<>
__device__ void dev_fun<int4>::store(volatile int4* p, unsigned int offset, const int4 &value){
	p += offset;
	// Streaming store
	asm volatile ("st.cs.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(p), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w) );
}
template<>
__device__ int dev_fun<int4>::first_element(const int4 &v){
	return v.x;
}
template<>
__device__ int dev_fun<int4>::reduce(const int4 &v){
	return v.x ^ v.y ^ v.z ^ v.w;
}


template <class T, bool readonly, int blockdim, int stepwidth, int index_clamping>
__global__ void benchmark_func(T * const g_data){
	dev_fun<T> func;
	const int grid_data_width = stepwidth*gridDim.x*blockdim;

	// Thread block-wise striding
	int index = stepwidth*blockIdx.x*blockdim + threadIdx.x;
	index = index_clamping==0 ? index : index % index_clamping;
	const int stride = blockdim;

	unsigned int offset = index;
	T temp = func.init(0);
	for(int j=0; j<TOTAL_ITERATIONS; j+=UNROLL_ITERATIONS){
		// Pretend updating of offset in order to force repetitive loads
		offset = func(temp, offset);
#ifndef TEX_LOADS
		union {
			const T *ptr;
			int2 i;
		} g_data_load_ptr = { g_data+offset };
#endif
		/*volatile*/ T * const g_data_store_ptr = g_data+offset+grid_data_width;
#pragma unroll
		for(int i=0; i<UNROLL_ITERATIONS; i++){
			const unsigned int iteration_offset = (readonly ? i : i >> 1) % stepwidth;//readonly ? i % stepwidth : (i >> 1) % stepwidth;
			if( readonly || (i % 2 == 0) ){
#ifdef TEX_LOADS
				const T v = func.load(g_data, offset+iteration_offset*stride);
#else
				const T v = func.load(g_data_load_ptr.ptr, iteration_offset*stride);
#endif
				if( readonly ){
#ifdef TEX_LOADS
					// Pretend update of offset in order to force reloads
					offset ^= func.reduce(v);
#else
					// Pretend update of data pointer in order to force reloads
					g_data_load_ptr.i.x ^= func.reduce(v);
#endif
				}
				temp = v;
			} else
				func.store( g_data_store_ptr, iteration_offset*stride, temp );
		}
	}
	offset = func(temp, offset);
	if( offset != index ) // Does not occur
		*g_data = func.init(offset);
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

template<class datatype>
void runbench_warmup(datatype *cd, long size){
	const long reduced_grid_size = size/(UNROLL_ITERATIONS_MEM)/32;
	const int TOTAL_REDUCED_BLOCKS = reduced_grid_size/BLOCK_SIZE;

	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimReducedGrid(TOTAL_REDUCED_BLOCKS, 1, 1);

	benchmark_func< datatype, false, BLOCK_SIZE, 1, 256 ><<< dimReducedGrid, dimBlock >>>(cd);
	CUDA_SAFE_CALL( cudaGetLastError() );
	CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}

template<class datatype, bool readonly, int stepwidth, int index_clamping>
double runbench(int total_blocks, datatype *cd, long size, bool spreadsheet){
	const long compute_grid_size = total_blocks*BLOCK_SIZE;
	const long data_size = ((index_clamping==0) ? compute_grid_size : min((int)compute_grid_size, (int)index_clamping))*stepwidth;//*(2-readonly);

	const long long total_iterations = (long long)(TOTAL_ITERATIONS)*compute_grid_size;
	const long long computations = total_iterations*(sizeof(datatype)/sizeof(int));//(long long)(TOTAL_ITERATIONS)*compute_grid_size;
	const long long memoryoperations = total_iterations;//(long long)(TOTAL_ITERATIONS)*compute_grid_size;

	// Set device memory
	CUDA_SAFE_CALL( cudaMemset(cd, 0, size*sizeof(datatype)) );  // initialize to zeros

	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(total_blocks, 1, 1);

	cudaEvent_t start, stop;

	if(!spreadsheet){
		printf("\nElement size %d, Grid size: %ld threads, Data size: %ld elements (%ld bytes)\n", (int)sizeof(datatype), compute_grid_size, data_size, data_size*sizeof(datatype)/*size*//*, 32*(UNROLL_ITERATIONS-1)*sizeof(datatype)*/);
	}

	initializeEvents(&start, &stop);
	benchmark_func< datatype, readonly, BLOCK_SIZE, stepwidth, index_clamping ><<< dimGrid, dimBlock >>>(cd);
	float kernel_time = finalizeEvents(start, stop);
	double bandwidth = ((double)memoryoperations*sizeof(datatype))/kernel_time*1000./(1000.*1000.*1000.);

	if(!spreadsheet){
		printf("\tKernel execution time :    %10.3f msecs\n", kernel_time);
		printf("\t   Compute throughput :    %10.3f GIOPS\n", ((double)computations)/kernel_time*1000./(double)(1000*1000*1000));
		printf("\tMemory bandwidth\n");
		printf("\t               Total  :    %10.2f GB/sec\n", ((double)memoryoperations*sizeof(datatype))/kernel_time*1000./(1000.*1000.*1000.));
		printf("\t               Loads  :    %10.2f GB/sec\n", ((double)memoryoperations*sizeof(datatype))/kernel_time*1000./(1000.*1000.*1000.) * (readonly ? 1.0 : 0.5));
		printf("\t               Stores :    %10.2f GB/sec\n", ((double)memoryoperations*sizeof(datatype))/kernel_time*1000./(1000.*1000.*1000.) * (readonly ? 0.0 : 0.5));
	} else {
		int current_device;
		cudaDeviceProp deviceProp;
		CUDA_SAFE_CALL( cudaGetDevice(&current_device) );
		CUDA_SAFE_CALL( cudaGetDeviceProperties(&deviceProp, current_device) );
		printf("%12d;%9ld;%6d;%8d;%10ld;%8ld;%14.3f;%13.3f;%10.3f;%8.3f;%9.3f\n",
			(int)sizeof(datatype), compute_grid_size, stepwidth, index_clamping, data_size, data_size*sizeof(datatype),
			kernel_time, 
			((double)computations)/kernel_time*1000./(double)(1000*1000*1000),
			bandwidth,
			((double)memoryoperations)/kernel_time*1000./(1000.*1000.*1000.),
			((double)memoryoperations)/kernel_time*1000./(1000.*1000.*1000.) / (deviceProp.multiProcessorCount*deviceProp.clockRate/1000000.0));
	}
	return bandwidth;
}

template<class datatype, bool readonly>
double cachebenchGPU(double *c, long size, bool excel){
	// Construct grid size
	cudaDeviceProp deviceProp;
	int current_device;
	CUDA_SAFE_CALL( cudaGetDevice(&current_device) );
	CUDA_SAFE_CALL( cudaGetDeviceProperties(&deviceProp, current_device) );
	const int SM_count = deviceProp.multiProcessorCount;
	const int Threads_per_SM = deviceProp.maxThreadsPerMultiProcessor;
	const int BLOCKS_PER_SM = Threads_per_SM/BLOCK_SIZE;
	const int TOTAL_BLOCKS = BLOCKS_PER_SM * SM_count;

	datatype *cd;

	CUDA_SAFE_CALL( cudaMalloc((void**)&cd, size*sizeof(datatype)) );

	// Set device memory
	CUDA_SAFE_CALL( cudaMemset(cd, 0, size*sizeof(datatype)) );  // initialize to zeros

	// Bind textures to buffer
	cudaBindTexture(0, texdataI1, cd, size*sizeof(datatype));
	cudaBindTexture(0, texdataI2, cd, size*sizeof(datatype));
	cudaBindTexture(0, texdataI4, cd, size*sizeof(datatype));

	// Synchronize in order to wait for memory operations to finish
	CUDA_SAFE_CALL( cudaDeviceSynchronize() );

	runbench_warmup(cd, size);

	double peak_bw = 0.0;

	peak_bw = max( peak_bw, runbench<datatype, readonly,  1,  512>(TOTAL_BLOCKS, cd, size, excel) );
	peak_bw = max( peak_bw, runbench<datatype, readonly,  1, 1024>(TOTAL_BLOCKS, cd, size, excel) );
	peak_bw = max( peak_bw, runbench<datatype, readonly,  1, 2048>(TOTAL_BLOCKS, cd, size, excel) );
	peak_bw = max( peak_bw, runbench<datatype, readonly,  1, 4096>(TOTAL_BLOCKS, cd, size, excel) );
	peak_bw = max( peak_bw, runbench<datatype, readonly,  1, 8192>(TOTAL_BLOCKS, cd, size, excel) );
	peak_bw = max( peak_bw, runbench<datatype, readonly,  1,    0>(TOTAL_BLOCKS, cd, size, excel) );
	peak_bw = max( peak_bw, runbench<datatype, readonly,  2,    0>(TOTAL_BLOCKS, cd, size, excel) );
	peak_bw = max( peak_bw, runbench<datatype, readonly,  3,    0>(TOTAL_BLOCKS, cd, size, excel) );
	peak_bw = max( peak_bw, runbench<datatype, readonly,  4,    0>(TOTAL_BLOCKS, cd, size, excel) );
	peak_bw = max( peak_bw, runbench<datatype, readonly,  5,    0>(TOTAL_BLOCKS, cd, size, excel) );
	peak_bw = max( peak_bw, runbench<datatype, readonly,  6,    0>(TOTAL_BLOCKS, cd, size, excel) );
	peak_bw = max( peak_bw, runbench<datatype, readonly,  7,    0>(TOTAL_BLOCKS, cd, size, excel) );
	peak_bw = max( peak_bw, runbench<datatype, readonly,  8,    0>(TOTAL_BLOCKS, cd, size, excel) );
	peak_bw = max( peak_bw, runbench<datatype, readonly,  9,    0>(TOTAL_BLOCKS, cd, size, excel) );
	peak_bw = max( peak_bw, runbench<datatype, readonly, 10,    0>(TOTAL_BLOCKS, cd, size, excel) );
	peak_bw = max( peak_bw, runbench<datatype, readonly, 11,    0>(TOTAL_BLOCKS, cd, size, excel) );
	peak_bw = max( peak_bw, runbench<datatype, readonly, 12,    0>(TOTAL_BLOCKS, cd, size, excel) );
	peak_bw = max( peak_bw, runbench<datatype, readonly, 13,    0>(TOTAL_BLOCKS, cd, size, excel) );
	peak_bw = max( peak_bw, runbench<datatype, readonly, 14,    0>(TOTAL_BLOCKS, cd, size, excel) );
	peak_bw = max( peak_bw, runbench<datatype, readonly, 15,    0>(TOTAL_BLOCKS, cd, size, excel) );
	peak_bw = max( peak_bw, runbench<datatype, readonly, 16,    0>(TOTAL_BLOCKS, cd, size, excel) );
	if( readonly ){
		peak_bw = max( peak_bw, runbench<datatype, readonly, 18,    0>(TOTAL_BLOCKS, cd, size, excel) );
		peak_bw = max( peak_bw, runbench<datatype, readonly, 20,    0>(TOTAL_BLOCKS, cd, size, excel) );
		peak_bw = max( peak_bw, runbench<datatype, readonly, 22,    0>(TOTAL_BLOCKS, cd, size, excel) );
		peak_bw = max( peak_bw, runbench<datatype, readonly, 24,    0>(TOTAL_BLOCKS, cd, size, excel) );
		peak_bw = max( peak_bw, runbench<datatype, readonly, 28,    0>(TOTAL_BLOCKS, cd, size, excel) );
		peak_bw = max( peak_bw, runbench<datatype, readonly, 32,    0>(TOTAL_BLOCKS, cd, size, excel) );
		peak_bw = max( peak_bw, runbench<datatype, readonly, 40,    0>(TOTAL_BLOCKS, cd, size, excel) );
		peak_bw = max( peak_bw, runbench<datatype, readonly, 48,    0>(TOTAL_BLOCKS, cd, size, excel) );
		peak_bw = max( peak_bw, runbench<datatype, readonly, 56,    0>(TOTAL_BLOCKS, cd, size, excel) );
		peak_bw = max( peak_bw, runbench<datatype, readonly, 64,    0>(TOTAL_BLOCKS, cd, size, excel) );
	} else {
		peak_bw = max( peak_bw, runbench<datatype, readonly, 18,    0>(TOTAL_BLOCKS, cd, size, excel) );
		peak_bw = max( peak_bw, runbench<datatype, readonly, 19,    0>(TOTAL_BLOCKS, cd, size, excel) );
		peak_bw = max( peak_bw, runbench<datatype, readonly, 20,    0>(TOTAL_BLOCKS, cd, size, excel) );
		peak_bw = max( peak_bw, runbench<datatype, readonly, 21,    0>(TOTAL_BLOCKS, cd, size, excel) );
		peak_bw = max( peak_bw, runbench<datatype, readonly, 22,    0>(TOTAL_BLOCKS, cd, size, excel) );
		peak_bw = max( peak_bw, runbench<datatype, readonly, 24,    0>(TOTAL_BLOCKS, cd, size, excel) );
		peak_bw = max( peak_bw, runbench<datatype, readonly, 26,    0>(TOTAL_BLOCKS, cd, size, excel) );
		peak_bw = max( peak_bw, runbench<datatype, readonly, 28,    0>(TOTAL_BLOCKS, cd, size, excel) );
		peak_bw = max( peak_bw, runbench<datatype, readonly, 30,    0>(TOTAL_BLOCKS, cd, size, excel) );
		peak_bw = max( peak_bw, runbench<datatype, readonly, 32,    0>(TOTAL_BLOCKS, cd, size, excel) );
	}

	// Copy results back to host memory
	CUDA_SAFE_CALL( cudaMemcpy(c, cd, size*sizeof(datatype), cudaMemcpyDeviceToHost) );

	// Unbind textures
	cudaUnbindTexture(texdataI1);
	cudaUnbindTexture(texdataI2);
	cudaUnbindTexture(texdataI4);

	CUDA_SAFE_CALL( cudaFree(cd) );
	return peak_bw;
}

extern "C" void cachebenchGPU(double *c, long size, bool excel){
#ifdef L2_ONLY
	printf("Global cache benchmark (L2 cache)\n");
#else
#ifdef TEX_LOADS
	printf("Texture cache benchmark\n");
#else
	printf("Whole cache hierarchy benchmark (L1 & L2 caches)\n");
#endif
#endif

	printf("\nRead only benchmark\n");
	if( excel ){
		printf("EXCEL header:\n");
		printf("Element size;Grid size; Parameters;   ; Data size;        ;Execution time;Instr.thr/put;Memory b/w; Ops/sec;Ops/cycle\n");
		printf("     (bytes);(threads);(step);(idx/cl);(elements); (bytes);       (msecs);      (GIOPS);  (GB/sec);  (10^9);   per SM\n");
	}
	double peak_bw_ro_int1 = cachebenchGPU<int,  true>(c, size, excel);
	double peak_bw_ro_int2 = cachebenchGPU<int2, true>(c, size, excel);
	double peak_bw_ro_int4 = cachebenchGPU<int4, true>(c, size, excel);

	printf("\nRead-Write benchmark (*probably not reliable due to mixed L2-DRAM load-store transcations)\n");
	if( excel ){
		printf("EXCEL header:\n");
		printf("Element size;Grid size; Parameters;   ; Data size;        ;Execution time;Instr.thr/put;Memory b/w; Ops/sec;Ops/cycle\n");
		printf("     (bytes);(threads);(step);(idx/cl);(elements); (bytes);       (msecs);      (GIOPS);  (GB/sec);  (10^9);   per SM\n");
	}
	double peak_bw_rw_int1 = cachebenchGPU<int,  false>(c, size, excel);
	double peak_bw_rw_int2 = cachebenchGPU<int2, false>(c, size, excel);
	double peak_bw_rw_int4 = cachebenchGPU<int4, false>(c, size, excel);

	printf("\nPeak bandwidth measurements per element size and access type\n");
	printf("\tRead only accesses:\n");
	printf("\t\tint1: %10.2f GB/sec\n", peak_bw_ro_int1);
	printf("\t\tint2: %10.2f GB/sec\n", peak_bw_ro_int2);
	printf("\t\tint4: %10.2f GB/sec\n", peak_bw_ro_int4);
	printf("\t\tmax:  %10.2f GB/sec\n", max3(peak_bw_ro_int1, peak_bw_ro_int2, peak_bw_ro_int4));
	printf("\tRead-write accesses:\n");
	printf("\t\tint1: %10.2f GB/sec\n", peak_bw_rw_int1);
	printf("\t\tint2: %10.2f GB/sec\n", peak_bw_rw_int2);
	printf("\t\tint4: %10.2f GB/sec\n", peak_bw_rw_int4);
	printf("\t\tmax:  %10.2f GB/sec\n", max3(peak_bw_rw_int1, peak_bw_rw_int2, peak_bw_rw_int4));

	CUDA_SAFE_CALL( cudaDeviceReset() );
}
