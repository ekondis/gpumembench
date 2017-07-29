/**
 * const_kernels.cpp: This file is part of the gpumembench micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <stdio.h>
#include <loclutil.h>
#include "const_kernels.h"
#include "kernel.h"

#define TEXTIFY(a) TEXTIFY__(a)
#define TEXTIFY__(a) #a

extern "C" void constbenchGPU(cl_device_id selected_device_id, int *a, size_t gridsize){
	const size_t BLOCK_SIZE = 256;
	//const int TOTAL_BLOCKS = gridsize/(BLOCK_SIZE);
	int c;

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

	//char c_parameters[256];
	//sprintf(c_parameters, )
	cl_program program1 = loclBuildProgram(context, selected_device_id, c_kernel, "-cl-std=CL1.1 -DVECTOR_SIZE=" TEXTIFY(VECTOR_SIZE) " -D__CUSTOM_TYPE__=int");
	cl_program program2 = loclBuildProgram(context, selected_device_id, c_kernel, "-cl-std=CL1.1 -DVECTOR_SIZE=" TEXTIFY(VECTOR_SIZE) " -D__CUSTOM_TYPE__=int2");
	cl_program program3 = loclBuildProgram(context, selected_device_id, c_kernel, "-cl-std=CL1.1 -DVECTOR_SIZE=" TEXTIFY(VECTOR_SIZE) " -D__CUSTOM_TYPE__=int4");

	// Create kernels
	printf("Creating kernels...     ");
	cl_kernel kernel_int1 = clCreateKernel(program1, "krn_benchmark_constant_int", &errno);
	cl_kernel kernel_int2 = clCreateKernel(program2, "krn_benchmark_constant_int2", &errno);
	cl_kernel kernel_int4 = clCreateKernel(program3, "krn_benchmark_constant_int4", &errno);
	printf("Ok\n");

	cl_mem dev_bufferC = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, &errno);
	cl_mem dev_buffer_constant = clCreateBuffer(context, CL_MEM_READ_ONLY, VECTOR_SIZE*sizeof(int), NULL, &errno);
	OCL_SAFE_CALL(errno);

	// Copy data to device memory
	OCL_SAFE_CALL( clEnqueueWriteBuffer(cmd_queue, dev_buffer_constant, CL_TRUE, 0, VECTOR_SIZE*sizeof(int), a, 0, NULL, NULL) );
	//OCL_SAFE_CALL( cudaMemcpyToSymbol(constant_data, a, VECTOR_SIZE*sizeof(int), 0, cudaMemcpyHostToDevice) );
	const int zero = 0;
	OCL_SAFE_CALL( clEnqueueWriteBuffer(cmd_queue, dev_bufferC, CL_TRUE, 0, sizeof(int), &zero, 0, NULL, NULL) );
	//OCL_SAFE_CALL( cudaMemset(cd, 0, sizeof(int)) );  // initialize to zeros

	// Synchronize in order to wait for memory operations to finish
	OCL_SAFE_CALL( clFinish(cmd_queue) );

	cl_event ev;

	// warm up
	OCL_SAFE_CALL( clSetKernelArg(kernel_int1, 0, sizeof(cl_mem), &dev_bufferC) );
	OCL_SAFE_CALL( clSetKernelArg(kernel_int1, 1, sizeof(cl_mem), &dev_buffer_constant) );
	OCL_SAFE_CALL( clEnqueueNDRangeKernel(cmd_queue, kernel_int1, 1, NULL, &gridsize, &BLOCK_SIZE, 0, NULL, NULL) );
	OCL_SAFE_CALL( clFinish(cmd_queue) );

	OCL_SAFE_CALL( clEnqueueWriteBuffer(cmd_queue, dev_bufferC, CL_TRUE, 0, sizeof(int), &zero, 0, NULL, NULL) );
	OCL_SAFE_CALL( clSetKernelArg(kernel_int1, 0, sizeof(cl_mem), &dev_bufferC) );
	OCL_SAFE_CALL( clSetKernelArg(kernel_int1, 1, sizeof(cl_mem), &dev_buffer_constant) );
	OCL_SAFE_CALL( clEnqueueNDRangeKernel(cmd_queue, kernel_int1, 1, NULL, &gridsize, &BLOCK_SIZE, 0, NULL, &ev) );
	// Synchronize in order to wait for kernel to finish
	double krn_time_constant_32b = loclGetEventExecTimeAndRelease(ev)*1000.;

	// Copy results back to host memory and validate result
	OCL_SAFE_CALL( clEnqueueReadBuffer(cmd_queue, dev_bufferC, CL_TRUE, 0, sizeof(int), &c, 0, NULL, NULL) );
	if( c!=VECTOR_SIZE ){
		fprintf(stderr, "ERROR: c was not %d (%d)!\n", VECTOR_SIZE, c);
		exit(1);
	}

	OCL_SAFE_CALL( clEnqueueWriteBuffer(cmd_queue, dev_bufferC, CL_TRUE, 0, sizeof(int), &zero, 0, NULL, NULL) );
	OCL_SAFE_CALL( clSetKernelArg(kernel_int2, 0, sizeof(cl_mem), &dev_bufferC) );
	OCL_SAFE_CALL( clSetKernelArg(kernel_int2, 1, sizeof(cl_mem), &dev_buffer_constant) );
	OCL_SAFE_CALL( clEnqueueNDRangeKernel(cmd_queue, kernel_int2, 1, NULL, &gridsize, &BLOCK_SIZE, 0, NULL, &ev) );
	double krn_time_constant_64b = loclGetEventExecTimeAndRelease(ev)*1000.;

	OCL_SAFE_CALL( clEnqueueReadBuffer(cmd_queue, dev_bufferC, CL_TRUE, 0, sizeof(int), &c, 0, NULL, NULL) );
	if( c!=VECTOR_SIZE ){
		fprintf(stderr, "ERROR: c was not %d (%d)!\n", VECTOR_SIZE, c);
		exit(1);
	}

	OCL_SAFE_CALL( clEnqueueWriteBuffer(cmd_queue, dev_bufferC, CL_TRUE, 0, sizeof(int), &zero, 0, NULL, NULL) );
	OCL_SAFE_CALL( clSetKernelArg(kernel_int4, 0, sizeof(cl_mem), &dev_bufferC) );
	OCL_SAFE_CALL( clSetKernelArg(kernel_int4, 1, sizeof(cl_mem), &dev_buffer_constant) );
	OCL_SAFE_CALL( clEnqueueNDRangeKernel(cmd_queue, kernel_int4, 1, NULL, &gridsize, &BLOCK_SIZE, 0, NULL, &ev) );
	double krn_time_constant_128b = loclGetEventExecTimeAndRelease(ev)*1000.;

	OCL_SAFE_CALL( clEnqueueReadBuffer(cmd_queue, dev_bufferC, CL_TRUE, 0, sizeof(int), &c, 0, NULL, NULL) );
	if( c!=VECTOR_SIZE ){
		fprintf(stderr, "ERROR: c was not %d (%d)!\n", VECTOR_SIZE, c);
		exit(1);
	}

	printf("Kernel execution time\n");
	printf("\tbenchmark_constant  (32bit):%10.4f msecs\n", krn_time_constant_32b);
	printf("\tbenchmark_constant  (64bit):%10.4f msecs\n", krn_time_constant_64b);
	printf("\tbenchmark_constant (128bit):%10.4f msecs\n", krn_time_constant_128b);
	
	printf("Total operations executed\n");
	const long long operations_bytes  = (long long)((VECTOR_SIZE)*gridsize*sizeof(int));
	const long long operations_32bit  = (long long)((VECTOR_SIZE)*gridsize/1);
	const long long operations_64bit  = (long long)((VECTOR_SIZE)*gridsize/2);
	const long long operations_128bit = (long long)((VECTOR_SIZE)*gridsize/4);
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
	printf("\tConstant memory operations per clock (32bit) :%8.2f (per SM%6.2f)\n",((double)operations_32bit)/(1000.*dev_freq*krn_time_constant_32b), ((double)operations_32bit)/(1000.*dev_freq*krn_time_constant_32b)/cnt_cunits);
	printf("\tConstant memory operations per clock (64bit) :%8.2f (per SM%6.2f)\n",((double)operations_64bit)/(1000.*dev_freq*krn_time_constant_64b), ((double)operations_64bit)/(1000.*dev_freq*krn_time_constant_64b)/cnt_cunits);
	printf("\tConstant memory operations per clock (128bit):%8.2f (per SM%6.2f)\n",((double)operations_128bit)/(1000.*dev_freq*krn_time_constant_128b), ((double)operations_128bit)/(1000.*dev_freq*krn_time_constant_128b)/cnt_cunits);

	printf("Compute overhead\n");
	printf("\tAddition operations per constant memory operation  (32bit): 1\n");
	printf("\tAddition operations per constant memory operation  (64bit): 2\n");
	printf("\tAddition operations per constant memory operation (128bit): 4\n");
	
	OCL_SAFE_CALL( clFinish(cmd_queue) );

	// Release program and kernels
	OCL_SAFE_CALL( clReleaseKernel(kernel_int1) );
	OCL_SAFE_CALL( clReleaseKernel(kernel_int2) );
	OCL_SAFE_CALL( clReleaseKernel(kernel_int4) );
	OCL_SAFE_CALL( clReleaseProgram(program1) );
	OCL_SAFE_CALL( clReleaseProgram(program2) );
	OCL_SAFE_CALL( clReleaseProgram(program3) );

	// Release command queue
	OCL_SAFE_CALL( clReleaseCommandQueue(cmd_queue) );

	// Release buffers
	OCL_SAFE_CALL( clReleaseMemObject(dev_buffer_constant) );
	OCL_SAFE_CALL( clReleaseMemObject(dev_bufferC) );

	// Release context
	OCL_SAFE_CALL( clReleaseContext(context) );
}
