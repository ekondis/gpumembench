/**
 * cache_kernels.cpp: This file is part of the gpumembench micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define _CRT_SECURE_NO_WARNINGS
#include <string>
#include <sstream>
#include <iostream>
#include <loclutil.h>
#include <cstdarg>
#include "cache_kernels.h"
#include "kernel.h"

#define TEXTIFY(a) TEXTIFY__(a)
#define TEXTIFY__(a) #a

#define TOTAL_ITERATIONS  (8192)
#define UNROLL_ITERATIONS (64)

#define UNROLL_ITERATIONS_MEM (UNROLL_ITERATIONS/2)

const int BLOCK_SIZE = 256;
const int IMAGE_ROW_LEN = 1024;//512;//4096;

typedef struct{int stepwidth, index_clamping;} tkernel_configuration;
const tkernel_configuration get_kernel_conf(bool readonly, int i){
	const tkernel_configuration KERNEL_CONF[] = {
		{1, 512},
		{1, 1024},
		{1, 2048},
		{1, 4096},
		{1, 8192},
		{1, 0},
		{2, 0},
		{3, 0},
		{4, 0},
		{5, 0},
		{6, 0},
		{7, 0},
		{8, 0},
		{9, 0},
		{10, 0},
		{11, 0},
		{12, 0},
		{13, 0},
		{14, 0},
		{15, 0},
		{16, 0},
		{18, 0},
		{readonly ? 20 : 19, 0},
		{readonly ? 22 : 20, 0},
		{readonly ? 24 : 21, 0},
		{readonly ? 28 : 22, 0},
		{readonly ? 32 : 24, 0},
		{readonly ? 40 : 26, 0},
		{readonly ? 48 : 28, 0},
		{readonly ? 56 : 30, 0},
		{readonly ? 64 : 32, 0},
	};
	if( i<0 ){
		tkernel_configuration count = {sizeof(KERNEL_CONF)/sizeof(tkernel_configuration), -1};
		return count;
	}
	return KERNEL_CONF[i];
}

double max3(double v1, double v2, double v3){
	double t = v1>v2 ? v1 : v2;
	return t>v3 ? t : v3;
}

template<class D>
D max(D v1, D v2){ return v1>v2 ? v1 : v2; }

template<class D>
D min(D v1, D v2){ return v1<v2 ? v1 : v2; }

void clbuffer_memset(cl_kernel kernel, cl_command_queue queue, cl_mem buffer, cl_uint len, cl_uint v){
//	OCL_SAFE_CALL( clEnqueueFillBuffer(queue, buffer, &v, sizeof(v), 0, len*sizeof(cl_uint), 0, NULL, NULL) );
	const size_t dimBlock(BLOCK_SIZE);
	const size_t dimGrid(64*BLOCK_SIZE);
	OCL_SAFE_CALL( clSetKernelArg(kernel, 0, sizeof(cl_uint), &len) );
	OCL_SAFE_CALL( clSetKernelArg(kernel, 1, sizeof(cl_uint), &v) );
	OCL_SAFE_CALL( clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer) );
	OCL_SAFE_CALL( clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &dimGrid, &dimBlock, 0, NULL, NULL) );
	OCL_SAFE_CALL( clFinish(queue) );
}

void flushed_printf(const char* format, ...){
	va_list args;
	va_start(args, format);
	vprintf(format, args);
	va_end(args);
	fflush(stdout);
}

void runbench_warmup(const size_t data_vecwidth, cl_mem cd, cl_mem c_tex, size_t size, cl_command_queue queue, cl_kernel kernel){
	const long reduced_grid_size = size/(UNROLL_ITERATIONS_MEM)/32;
	const int TOTAL_REDUCED_BLOCKS = reduced_grid_size/BLOCK_SIZE;

	const size_t dimBlock(BLOCK_SIZE);
	const size_t dimReducedGrid(TOTAL_REDUCED_BLOCKS*BLOCK_SIZE);

	// warm up
	OCL_SAFE_CALL( clSetKernelArg(kernel, 0, sizeof(cl_mem), &cd) );
	if( c_tex != NULL )
		OCL_SAFE_CALL( clSetKernelArg(kernel, 1, sizeof(cl_mem), &c_tex) );
	OCL_SAFE_CALL( clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &dimReducedGrid, &dimBlock, 0, NULL, NULL) );
	OCL_SAFE_CALL( clFinish(queue) );
}

double runbench(const size_t data_vecwidth, bool readonly, int total_blocks, cl_mem cd, cl_mem c_tex, size_t size, cl_command_queue queue, cl_kernel kernel, cl_kernel fill_kernel, bool spreadsheet, int stepwidth, int index_clamping){
	const long compute_grid_size = total_blocks*BLOCK_SIZE;
	const long data_size = ((index_clamping==0) ? compute_grid_size : min((int)compute_grid_size, (int)index_clamping))*stepwidth;

	const long long total_iterations = (long long)(TOTAL_ITERATIONS)*compute_grid_size;
	const long long computations = total_iterations*data_vecwidth;//(long long)(TOTAL_ITERATIONS)*compute_grid_size;
	const long long memoryoperations = total_iterations;//(long long)(TOTAL_ITERATIONS)*compute_grid_size;

	// Set device memory
	clbuffer_memset(fill_kernel, queue, cd, size, 0); // initialize to zeros

	const size_t dimBlock(BLOCK_SIZE);
	const size_t dimGrid(total_blocks*BLOCK_SIZE);

	cl_device_id selected_device_id;
	OCL_SAFE_CALL( clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &selected_device_id, NULL) );
	cl_uint SM_count, SM_clock;
	OCL_SAFE_CALL( clGetDeviceInfo(selected_device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(SM_count), &SM_count, NULL) );
	OCL_SAFE_CALL( clGetDeviceInfo(selected_device_id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(SM_clock), &SM_clock, NULL) );

	if(!spreadsheet){
		printf("\nElement size %d, Grid size: %ld threads, Data size: %ld elements (%ld bytes)\n", (int)(data_vecwidth*sizeof(int)), compute_grid_size, data_size, data_size*(data_vecwidth*sizeof(int)));
	}

	cl_event ev;

	OCL_SAFE_CALL( clSetKernelArg(kernel, 0, sizeof(cl_mem), &cd) );
	if( c_tex != NULL )
		OCL_SAFE_CALL( clSetKernelArg(kernel, 1, sizeof(cl_mem), &c_tex) );
	OCL_SAFE_CALL( clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &dimGrid, &dimBlock, 0, NULL, &ev) );
	// Synchronize in order to wait for kernel to finish
	double kernel_time = loclGetEventExecTimeAndRelease(ev)*1000.;
	double bandwidth = ((double)memoryoperations*(data_vecwidth*sizeof(int)))/kernel_time*1000./(1000.*1000.*1000.);

	if(!spreadsheet){
		printf("\tKernel execution time :    %10.3f msecs\n", kernel_time);
		printf("\t   Compute throughput :    %10.3f GIOPS\n", ((double)computations)/kernel_time*1000./(double)(1000*1000*1000));
		printf("\tMemory bandwidth\n");
		printf("\t               Total  :    %10.2f GB/sec\n", ((double)memoryoperations*(data_vecwidth*sizeof(int)))/kernel_time*1000./(1000.*1000.*1000.));
		printf("\t               Loads  :    %10.2f GB/sec\n", ((double)memoryoperations*(data_vecwidth*sizeof(int)))/kernel_time*1000./(1000.*1000.*1000.) * (readonly ? 1.0 : 0.5));
		printf("\t               Stores :    %10.2f GB/sec\n", ((double)memoryoperations*(data_vecwidth*sizeof(int)))/kernel_time*1000./(1000.*1000.*1000.) * (readonly ? 0.0 : 0.5));
	} else {
		printf("%12d;%9ld;%6d;%8d;%10ld;%8ld;%14.3f;%13.3f;%10.3f;%8.3f;%9.3f\n",
			(int)(data_vecwidth*sizeof(int)), compute_grid_size, stepwidth, index_clamping, data_size, data_size*(data_vecwidth*sizeof(int)),
			kernel_time, 
			((double)computations)/kernel_time*1000./(double)(1000*1000*1000),
			bandwidth,
			((double)memoryoperations)/kernel_time*1000./(1000.*1000.*1000.),
			((double)memoryoperations)/kernel_time*1000./(1000.*1000.*1000.) / (SM_count*SM_clock/1000000.0));
	}
	return bandwidth;
}

double cachebenchGPU(const size_t data_vecwidth, bool readonly, cl_command_queue queue, bench_type btype, double *c, size_t size, bool excel, cl_kernel kernels[], int kernel_cnt, cl_kernel kernel_fill){
	// Construct grid size
	cl_device_id selected_device_id;
	OCL_SAFE_CALL( clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &selected_device_id, NULL) );
	cl_uint SM_count;
	OCL_SAFE_CALL( clGetDeviceInfo(selected_device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(SM_count), &SM_count, NULL) );
	cl_context context;
	OCL_SAFE_CALL( clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, NULL) );

	const int Threads_per_SM = 2048;//???? 2560; //deviceProp.maxThreadsPerMultiProcessor;
	const int BLOCKS_PER_SM = Threads_per_SM/BLOCK_SIZE;
	const int TOTAL_BLOCKS = BLOCKS_PER_SM * SM_count;

	cl_int errno;
	cl_mem dev_bufferC = clCreateBuffer(context, CL_MEM_READ_WRITE, size*sizeof(int)*data_vecwidth, NULL, &errno);
	OCL_SAFE_CALL(errno);

	// Set device memory
	clbuffer_memset(kernel_fill, queue, dev_bufferC, size, 0); // initialize to zeros

	// Bind textures to buffer
	cl_mem dev_buffer_img = NULL;
	if( btype == btTexture ){
		cl_channel_order ch_o;
		switch( data_vecwidth ){
		case 1:
			ch_o = CL_R;
			break;
		case 2:
			ch_o = CL_RG;
			break;
		default:
			ch_o = CL_RGBA;
		}
		const cl_image_format format = { ch_o, CL_UNSIGNED_INT32 };
		const size_t zero_dims[3] = {0, 0, 0};
		const size_t image_reg[3] = {IMAGE_ROW_LEN, size/IMAGE_ROW_LEN/2, 1};
		dev_buffer_img = clCreateImage2D(context, CL_MEM_READ_ONLY, &format, image_reg[0], image_reg[1], 0, NULL, &errno);
		OCL_SAFE_CALL(errno);
		OCL_SAFE_CALL( clEnqueueCopyBufferToImage(queue, dev_bufferC, dev_buffer_img, 0, zero_dims, image_reg, 0, NULL, NULL) );
	}

	// Synchronize in order to wait for memory operations to finish
	OCL_SAFE_CALL( clFinish(queue) );

	runbench_warmup(data_vecwidth, dev_bufferC, dev_buffer_img, size, queue, kernels[0]);

	double peak_bw = 0.0;

	for(int kernel_idx=0; kernel_idx<get_kernel_conf(readonly, -1).stepwidth; kernel_idx++)
		peak_bw = max( peak_bw, runbench(data_vecwidth, readonly, TOTAL_BLOCKS, dev_bufferC, dev_buffer_img, size, queue, kernels[kernel_idx], kernel_fill, excel, get_kernel_conf(readonly, kernel_idx).stepwidth,  get_kernel_conf(readonly, kernel_idx).index_clamping) );

	// Copy results back to host memory
	OCL_SAFE_CALL( clEnqueueReadBuffer(queue, dev_bufferC, CL_TRUE, 0, size*sizeof(int)*data_vecwidth, c, 0, NULL, NULL) );

	if( btype == btTexture )
		OCL_SAFE_CALL( clReleaseMemObject(dev_buffer_img) );

	OCL_SAFE_CALL( clReleaseMemObject(dev_bufferC) );
	return peak_bw;
}

void build_kernel(cl_context context, cl_device_id device_id, const char *c_kernel, const char *c_options, int stepwidth, int index_clamping, cl_program &program, cl_kernel &kernel){
	char c_options_tmp[256];
	sprintf(c_options_tmp, "%s -DSTEPWIDTH=%d -DINDEX_CLAMPING=%d", c_options, stepwidth, index_clamping);
	program = loclBuildProgram(context, device_id, c_kernel, c_options_tmp, false);
	cl_int errno;
	kernel = clCreateKernel(program, "benchmark_func", &errno);
	OCL_SAFE_CALL(errno);
	flushed_printf(".");
}

void build_kernels(cl_context context, cl_device_id device_id, const char *c_kernel, bench_type btype, int veclen, bool readonly, cl_program programs[], cl_kernel kernels[], int &count, cl_kernel &fill_kernel){
	count = 0;
	std::string s;
	std::ostringstream build_options;
	build_options << "-cl-std=CL1.1 -DVECTOR_SIZE=" TEXTIFY(VECTOR_SIZE) " -D__CUSTOM_TYPE__=int";
//	std::cout << build_options.str() <<std::endl;
	
	if( veclen>1 )
		build_options << veclen;

	build_options << " -DBLOCK_SIZE=" << BLOCK_SIZE;
	build_options << " -DTOTAL_ITERATIONS=" << TOTAL_ITERATIONS;
	build_options << " -DUNROLL_ITERATIONS=" << UNROLL_ITERATIONS;

	if( readonly )
		build_options << " -DREADONLY";
	if( btype == btL2Cache )
		build_options << " -DVOL_LOADS";
	if( btype == btTexture )
		build_options << " -DTEX_LOADS -DIMAGE_ROW_LEN=" << IMAGE_ROW_LEN;
//	std::cout << build_options.str();

	flushed_printf("Building kernels [");
	for(unsigned int i=0; i<static_cast<unsigned int>(get_kernel_conf(readonly, -1).stepwidth); i++){
		build_kernel(context, device_id, c_kernel, build_options.str().c_str(), get_kernel_conf(readonly, i).stepwidth, get_kernel_conf(readonly, i).index_clamping, programs[count], kernels[count]);
		count++;
		if( i==0 ){
			cl_int errno;
			fill_kernel = clCreateKernel(programs[0], "fill_buffer", &errno);
			OCL_SAFE_CALL( errno );
		}
	}
	flushed_printf("]\n");
}

void dispose_kernels(cl_program programs[], cl_kernel kernels[], int &count, cl_kernel fill_kernel){
	// Release program and kernels
	OCL_SAFE_CALL( clReleaseKernel(fill_kernel) );
	for(int i=0; i<count; i++){
		OCL_SAFE_CALL( clReleaseKernel(kernels[i]) );
		OCL_SAFE_CALL( clReleaseProgram(programs[i]) );
	}
	count = 0;
}

extern "C" void cachebenchGPU(cl_device_id selected_device_id, double *c, size_t gridsize, bench_type btype, bool excel){
	// Get platform ID
	cl_platform_id platform_id;
	OCL_SAFE_CALL( clGetDeviceInfo(selected_device_id, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &platform_id, NULL) );
	// Set context properties
	cl_context_properties ctxProps[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id, 0 };
	
	// Get #CUs and Clock frequency
	cl_uint cnt_cunits, dev_freq;
	OCL_SAFE_CALL( clGetDeviceInfo(selected_device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cnt_cunits), &cnt_cunits, NULL) );
	OCL_SAFE_CALL( clGetDeviceInfo(selected_device_id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(dev_freq), &dev_freq, NULL) );
	printf("Total CUs: %d\nMaximum clock frequency: %dMHz\n", cnt_cunits, dev_freq);

	cl_int errno;
	// Create context
	cl_context context = clCreateContext(ctxProps, 1, &selected_device_id, NULL, NULL, &errno);
	OCL_SAFE_CALL(errno);

	// Create command queue
	cl_command_queue cmd_queue = clCreateCommandQueue(context, selected_device_id, CL_QUEUE_PROFILING_ENABLE, &errno);
	OCL_SAFE_CALL(errno);

	switch(btype){
		case btAllCache:
			printf("Whole cache hierarchy benchmark (L1 & L2 caches)\n");
			break;
		case btL2Cache:
			printf("Global cache benchmark (L2 cache)\n");
			break;
		case btTexture:
			printf("Texture cache benchmark\n");
	}

	printf("\nRead only benchmark\n");
	if( excel ){
		printf("EXCEL header:\n");
		printf("Element size;Grid size; Parameters;   ; Data size;        ;Execution time;Instr.thr/put;Memory b/w; Ops/sec;Ops/cycle\n");
		printf("     (bytes);(threads);(step);(idx/cl);(elements); (bytes);       (msecs);      (GIOPS);  (GB/sec);  (10^9);   per SM\n");
	}

	cl_program programs[64];
	cl_kernel kernels[64];
	cl_kernel fill_kernel;
	int kernel_count = 0;

	build_kernels(context, selected_device_id, c_kernel, btype, 1, true, programs, kernels, kernel_count, fill_kernel);
	double peak_bw_ro_int1 = cachebenchGPU(1, true, cmd_queue, btype, c, gridsize, excel, kernels, kernel_count, fill_kernel);
	dispose_kernels(programs, kernels, kernel_count, fill_kernel);

	build_kernels(context, selected_device_id, c_kernel, btype, 2, true, programs, kernels, kernel_count, fill_kernel);
	double peak_bw_ro_int2 = cachebenchGPU(2, true, cmd_queue, btype, c, gridsize, excel, kernels, kernel_count, fill_kernel);
	dispose_kernels(programs, kernels, kernel_count, fill_kernel);

	build_kernels(context, selected_device_id, c_kernel, btype, 4, true, programs, kernels, kernel_count, fill_kernel);
	double peak_bw_ro_int4 = cachebenchGPU(4, true, cmd_queue, btype, c, gridsize, excel, kernels, kernel_count, fill_kernel);
	dispose_kernels(programs, kernels, kernel_count, fill_kernel);

	printf("\nRead-Write benchmark (*probably not reliable due to mixed L2-DRAM load-store transcations)\n");
	if( excel ){
		printf("EXCEL header:\n");
		printf("Element size;Grid size; Parameters;   ; Data size;        ;Execution time;Instr.thr/put;Memory b/w; Ops/sec;Ops/cycle\n");
		printf("     (bytes);(threads);(step);(idx/cl);(elements); (bytes);       (msecs);      (GIOPS);  (GB/sec);  (10^9);   per SM\n");
	}

	build_kernels(context, selected_device_id, c_kernel, btype, 1, false, programs, kernels, kernel_count, fill_kernel);
	double peak_bw_rw_int1 = cachebenchGPU(1, false, cmd_queue, btype, c, gridsize, excel, kernels, kernel_count, fill_kernel);
	dispose_kernels(programs, kernels, kernel_count, fill_kernel);

	build_kernels(context, selected_device_id, c_kernel, btype, 2, false, programs, kernels, kernel_count, fill_kernel);
	double peak_bw_rw_int2 = cachebenchGPU(2, false, cmd_queue, btype, c, gridsize, excel, kernels, kernel_count, fill_kernel);
	dispose_kernels(programs, kernels, kernel_count, fill_kernel);

	build_kernels(context, selected_device_id, c_kernel, btype, 4, false, programs, kernels, kernel_count, fill_kernel);
	double peak_bw_rw_int4 = cachebenchGPU(4, false, cmd_queue, btype, c, gridsize, excel, kernels, kernel_count, fill_kernel);
	dispose_kernels(programs, kernels, kernel_count, fill_kernel);

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

	// Release command queue
	OCL_SAFE_CALL( clReleaseCommandQueue(cmd_queue) );

	// Release context
	OCL_SAFE_CALL( clReleaseContext(context) );
}
