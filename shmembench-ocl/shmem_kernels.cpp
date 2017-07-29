/**
 * shmem_kernels.cpp: This file is part of the gpumembench micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <stdio.h>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <loclutil.h>
#include "kernel.h"

#define TOTAL_ITERATIONS (1024)

double max3(double v1, double v2, double v3){
	double t = v1>v2 ? v1 : v2;
	return t>v3 ? t : v3;
}

extern "C" void shmembenchGPU(cl_device_id selected_device_id, int *c, long size){
	const size_t BLOCK_SIZE = 256;//128;//256;

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

	cl_program program1 = loclBuildProgram(context, selected_device_id, c_kernel, "-cl-std=CL1.1 -D__CUSTOM_TYPE__=int");
	cl_program program2 = loclBuildProgram(context, selected_device_id, c_kernel, "-cl-std=CL1.1 -D__CUSTOM_TYPE__=int2");
	cl_program program3 = loclBuildProgram(context, selected_device_id, c_kernel, "-cl-std=CL1.1 -D__CUSTOM_TYPE__=int4");

	// Create kernels
	printf("Creating kernels...     ");
	cl_kernel kernel_int1 = clCreateKernel(program1, "krn_localmem_juggling_int", &errno);
	cl_kernel kernel_int2 = clCreateKernel(program2, "krn_localmem_juggling_int2", &errno);
	cl_kernel kernel_int4 = clCreateKernel(program3, "krn_localmem_juggling_int4", &errno);
	printf("Ok\n");

	printf("Creating buffer...      ");
	size_t buffer_len = size * sizeof(int);
	cl_mem dev_bufferC = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_len, NULL, &errno);
	OCL_SAFE_CALL(errno);
	printf("Ok (allocated %uMB)\n", (unsigned int)buffer_len/(1024*1024));

	// Copy data to device memory
	OCL_SAFE_CALL( clEnqueueWriteBuffer(cmd_queue, dev_bufferC, CL_TRUE, 0, buffer_len, c, 0, NULL, NULL) );

	// Synchronize in order to wait for memory operations to finish
	OCL_SAFE_CALL( clFinish(cmd_queue) );

	size_t global_worksize_int = size;
	size_t global_worksize_int2 = size/2;
	size_t global_worksize_int4 = size/4;
	cl_event ev;

	// warm up
	OCL_SAFE_CALL( clSetKernelArg(kernel_int1, 0, sizeof(cl_mem), &dev_bufferC) );
	OCL_SAFE_CALL( clSetKernelArg(kernel_int1, 1, sizeof(cl_int)*6*BLOCK_SIZE, NULL) );
	OCL_SAFE_CALL( clEnqueueNDRangeKernel(cmd_queue, kernel_int1, 1, NULL, &global_worksize_int4, &BLOCK_SIZE, 0, NULL, NULL) );
	OCL_SAFE_CALL( clFinish(cmd_queue) );

	OCL_SAFE_CALL( clSetKernelArg(kernel_int1, 0, sizeof(cl_mem), &dev_bufferC) );
	OCL_SAFE_CALL( clSetKernelArg(kernel_int1, 1, sizeof(cl_int)*6*BLOCK_SIZE, NULL) );
	OCL_SAFE_CALL( clEnqueueNDRangeKernel(cmd_queue, kernel_int1, 1, NULL, &global_worksize_int, &BLOCK_SIZE, 0, NULL, &ev) );

	// Synchronize in order to wait for kernel to finish
	double krn_time_shmem_32b = loclGetEventExecTimeAndRelease(ev)*1000.;
	
	OCL_SAFE_CALL( clSetKernelArg(kernel_int2, 0, sizeof(cl_mem), &dev_bufferC) );
	OCL_SAFE_CALL( clSetKernelArg(kernel_int2, 1, sizeof(cl_int2)*6*BLOCK_SIZE, NULL) );
	OCL_SAFE_CALL( clEnqueueNDRangeKernel(cmd_queue, kernel_int2, 1, NULL, &global_worksize_int2, &BLOCK_SIZE, 0, NULL, &ev) );

	// Synchronize in order to wait for kernel to finish
	double krn_time_shmem_64b = loclGetEventExecTimeAndRelease(ev)*1000.;

	OCL_SAFE_CALL( clSetKernelArg(kernel_int4, 0, sizeof(cl_mem), &dev_bufferC) );
	OCL_SAFE_CALL( clSetKernelArg(kernel_int4, 1, sizeof(cl_int4)*6*BLOCK_SIZE, NULL) );
	OCL_SAFE_CALL( clEnqueueNDRangeKernel(cmd_queue, kernel_int4, 1, NULL, &global_worksize_int4, &BLOCK_SIZE, 0, NULL, &ev) );

	// Synchronize in order to wait for kernel to finish
	double krn_time_shmem_128b = loclGetEventExecTimeAndRelease(ev)*1000.;

	printf("Execution time: %f\n", krn_time_shmem_32b);

	printf("Kernel execution time\n");
	printf("\tbenchmark_shmem  (32bit):%10.3f msecs\n", krn_time_shmem_32b);
	printf("\tbenchmark_shmem  (64bit):%10.3f msecs\n", krn_time_shmem_64b);
	printf("\tbenchmark_shmem (128bit):%10.3f msecs\n", krn_time_shmem_128b);
	
	printf("Total operations executed\n");
	const long long operations_bytes  = (6LL+4*5*TOTAL_ITERATIONS+6)*size*sizeof(int);
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

	printf("Normalized per CU\n");
	printf("\tlocal memory operations per clock (32bit) :%8.2f (per CU%6.2f)\n",((double)operations_32bit)/(1000.*dev_freq*krn_time_shmem_32b), ((double)operations_32bit)/(1000.*dev_freq*krn_time_shmem_32b)/cnt_cunits);
	printf("\tlocal memory operations per clock (64bit) :%8.2f (per CU%6.2f)\n",((double)operations_64bit)/(1000.*dev_freq*krn_time_shmem_64b), ((double)operations_64bit)/(1000.*dev_freq*krn_time_shmem_64b)/cnt_cunits);
	printf("\tlocal memory operations per clock (128bit):%8.2f (per CU%6.2f)\n",((double)operations_128bit)/(1000.*dev_freq*krn_time_shmem_128b), ((double)operations_128bit)/(1000.*dev_freq*krn_time_shmem_128b)/cnt_cunits);

	// Release program and kernels
	OCL_SAFE_CALL( clReleaseKernel(kernel_int1) );
	OCL_SAFE_CALL( clReleaseKernel(kernel_int2) );
	OCL_SAFE_CALL( clReleaseKernel(kernel_int4) );
	OCL_SAFE_CALL( clReleaseProgram(program1) );
	OCL_SAFE_CALL( clReleaseProgram(program2) );
	OCL_SAFE_CALL( clReleaseProgram(program3) );

	// Release command queue
	OCL_SAFE_CALL( clReleaseCommandQueue(cmd_queue) );

	// Release buffer
	OCL_SAFE_CALL( clReleaseMemObject(dev_bufferC) );

	// Release context
	OCL_SAFE_CALL( clReleaseContext(context) );
}
