#ifndef _LOCLUTIL_H_
#define _LOCLUTIL_H_

#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

#include <CL/opencl.h>

inline const char* loclGetErrCode(cl_int err){
	switch(err){
		case CL_SUCCESS:
			return "CL_SUCCESS";
		case CL_DEVICE_NOT_FOUND:
			return "CL_DEVICE_NOT_FOUND";
		case CL_DEVICE_NOT_AVAILABLE:
			return "CL_DEVICE_NOT_AVAILABLE";
		case CL_COMPILER_NOT_AVAILABLE:
			return "CL_COMPILER_NOT_AVAILABLE";
		case CL_MEM_OBJECT_ALLOCATION_FAILURE:
			return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
		case CL_OUT_OF_RESOURCES:
			return "CL_OUT_OF_RESOURCES";
		case CL_OUT_OF_HOST_MEMORY:
			return "CL_OUT_OF_HOST_MEMORY";
		case CL_PROFILING_INFO_NOT_AVAILABLE:
			return "CL_PROFILING_INFO_NOT_AVAILABLE";
		case CL_MEM_COPY_OVERLAP:
			return "CL_MEM_COPY_OVERLAP";
		case CL_IMAGE_FORMAT_MISMATCH:
			return "CL_IMAGE_FORMAT_MISMATCH";
		case CL_IMAGE_FORMAT_NOT_SUPPORTED:
			return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
		case CL_BUILD_PROGRAM_FAILURE:
			return "CL_BUILD_PROGRAM_FAILURE";
		case CL_MAP_FAILURE:
			return "CL_MAP_FAILURE";
		case CL_MISALIGNED_SUB_BUFFER_OFFSET:
			return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
		case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
			return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
		case CL_INVALID_VALUE:
			return "CL_INVALID_VALUE";
		case CL_INVALID_DEVICE_TYPE:
			return "CL_INVALID_DEVICE_TYPE";
		case CL_INVALID_PLATFORM:
			return "CL_INVALID_PLATFORM";
		case CL_INVALID_DEVICE:
			return "CL_INVALID_DEVICE";
		case CL_INVALID_CONTEXT:
			return "CL_INVALID_CONTEXT";
		case CL_INVALID_QUEUE_PROPERTIES:
			return "CL_INVALID_QUEUE_PROPERTIES";
		case CL_INVALID_COMMAND_QUEUE:
			return "CL_INVALID_COMMAND_QUEUE";
		case CL_INVALID_HOST_PTR:
			return "CL_INVALID_HOST_PTR";
		case CL_INVALID_MEM_OBJECT:
			return "CL_INVALID_MEM_OBJECT";
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
			return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
		case CL_INVALID_IMAGE_SIZE:
			return "CL_INVALID_IMAGE_SIZE";
		case CL_INVALID_SAMPLER:
			return "CL_INVALID_SAMPLER";
		case CL_INVALID_BINARY:
			return "CL_INVALID_BINARY";
		case CL_INVALID_BUILD_OPTIONS:
			return "CL_INVALID_BUILD_OPTIONS";
		case CL_INVALID_PROGRAM:
			return "CL_INVALID_PROGRAM";
		case CL_INVALID_PROGRAM_EXECUTABLE:
			return "CL_INVALID_PROGRAM_EXECUTABLE";
		case CL_INVALID_KERNEL_NAME:
			return "CL_INVALID_KERNEL_NAME";
		case CL_INVALID_KERNEL_DEFINITION:
			return "CL_INVALID_KERNEL_DEFINITION";
		case CL_INVALID_KERNEL:
			return "CL_INVALID_KERNEL";
		case CL_INVALID_ARG_INDEX:
			return "CL_INVALID_ARG_INDEX";
		case CL_INVALID_ARG_VALUE:
			return "CL_INVALID_ARG_VALUE";
		case CL_INVALID_ARG_SIZE:
			return "CL_INVALID_ARG_SIZE";
		case CL_INVALID_KERNEL_ARGS:
			return "CL_INVALID_KERNEL_ARGS";
		case CL_INVALID_WORK_DIMENSION:
			return "CL_INVALID_WORK_DIMENSION";
		case CL_INVALID_WORK_GROUP_SIZE:
			return "CL_INVALID_WORK_GROUP_SIZE";
		case CL_INVALID_WORK_ITEM_SIZE:
			return "CL_INVALID_WORK_ITEM_SIZE";
		case CL_INVALID_GLOBAL_OFFSET:
			return "CL_INVALID_GLOBAL_OFFSET";
		case CL_INVALID_EVENT_WAIT_LIST:
			return "CL_INVALID_EVENT_WAIT_LIST";
		case CL_INVALID_EVENT:
			return "CL_INVALID_EVENT";
		case CL_INVALID_OPERATION:
			return "CL_INVALID_OPERATION";
		case CL_INVALID_GL_OBJECT:
			return "CL_INVALID_GL_OBJECT";
		case CL_INVALID_BUFFER_SIZE:
			return "CL_INVALID_BUFFER_SIZE";
		case CL_INVALID_MIP_LEVEL:
			return "CL_INVALID_MIP_LEVEL";
		case CL_INVALID_GLOBAL_WORK_SIZE:
			return "CL_INVALID_GLOBAL_WORK_SIZE";
		case CL_INVALID_PROPERTY:
			return "CL_INVALID_PROPERTY";
	}
	return "unknown";
}
#  define OCL_SAFE_CALL(call) {                                    \
    cl_int err = call;                                                    \
    if( CL_SUCCESS != err) {                                                \
        fprintf(stderr, "OpenCL error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, loclGetErrCode(err) );              \
        exit(EXIT_FAILURE);                                                  \
    } }

#define FRACTION_CEILING(numerator, denominator) ((numerator+denominator-1)/(denominator))

inline void loclPrintAvailableDevices(void){
	cl_uint cnt_platforms, cnt_device_ids;
	cl_platform_id *platform_ids;
	OCL_SAFE_CALL( clGetPlatformIDs(0, NULL, &cnt_platforms) );

	platform_ids = (cl_platform_id*)alloca(sizeof(cl_platform_id)*cnt_platforms);
	OCL_SAFE_CALL( clGetPlatformIDs(cnt_platforms, platform_ids, NULL) );

	int cur_dev_idx = 1;
	for(int i=0; i<(int)cnt_platforms; i++){
		OCL_SAFE_CALL( clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, 0, NULL, &cnt_device_ids) );

		size_t t;
		OCL_SAFE_CALL( clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME, 0, NULL, &t) );
		char *cl_plf_name = (char*)alloca( t );
		OCL_SAFE_CALL( clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME, t, cl_plf_name, NULL) );
		OCL_SAFE_CALL( clGetPlatformInfo(platform_ids[i], CL_PLATFORM_VENDOR, 0, NULL, &t) );
		char *cl_plf_vendor = (char*)alloca( t );
		OCL_SAFE_CALL( clGetPlatformInfo(platform_ids[i], CL_PLATFORM_VENDOR, t, cl_plf_vendor, NULL) );

		printf("Platform %d (name:%s, vendor:%s) has %d devices\n", i+1, cl_plf_name, cl_plf_vendor, (int)cnt_device_ids);

		cl_device_id *device_ids = (cl_device_id*)alloca(sizeof(cl_device_id)*cnt_device_ids);
		OCL_SAFE_CALL( clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, cnt_device_ids, device_ids, NULL) );

		for(int d=0; d<(int)cnt_device_ids; d++){
			OCL_SAFE_CALL( clGetDeviceInfo(device_ids[d], CL_DEVICE_NAME, 0, NULL, &t) );
			char *cl_dev_name = (char*)alloca( t );
			OCL_SAFE_CALL( clGetDeviceInfo(device_ids[d], CL_DEVICE_NAME, t, cl_dev_name, NULL) );
			OCL_SAFE_CALL( clGetDeviceInfo(device_ids[d], CL_DEVICE_VENDOR, 0, NULL, &t) );
			char *cl_dev_vendor = (char*)alloca( t );
			OCL_SAFE_CALL( clGetDeviceInfo(device_ids[d], CL_DEVICE_VENDOR, t, cl_dev_vendor, NULL) );
			OCL_SAFE_CALL( clGetDeviceInfo(device_ids[d], CL_DRIVER_VERSION, 0, NULL, &t) );
			char *cl_driver_ver = (char*)alloca( t );
			OCL_SAFE_CALL( clGetDeviceInfo(device_ids[d], CL_DRIVER_VERSION, t, cl_driver_ver, NULL) );
			OCL_SAFE_CALL( clGetDeviceInfo(device_ids[d], CL_DEVICE_VERSION, 0, NULL, &t) );
			char *cl_device_ver = (char*)alloca( t );
			OCL_SAFE_CALL( clGetDeviceInfo(device_ids[d], CL_DEVICE_VERSION, t, cl_device_ver, NULL) );
			cl_uint cl_max_freq, cl_max_cunits;
			OCL_SAFE_CALL( clGetDeviceInfo(device_ids[d], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_max_freq), &cl_max_freq, NULL) );
			OCL_SAFE_CALL( clGetDeviceInfo(device_ids[d], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_max_cunits), &cl_max_cunits, NULL) );
			printf(" [%d].Vendor: %s Device: %s (%d compute units, freq %dMHz, %s, driver: %s)\n", cur_dev_idx, cl_dev_vendor, cl_dev_name, cl_max_cunits, cl_max_freq, cl_device_ver, cl_driver_ver);

			cur_dev_idx++;
		}
	}
	if( cur_dev_idx>1 )
		printf("Please select a device(1..%d)\n", cur_dev_idx-1);
	else
		printf("No OpenCL device found\n");
}

inline cl_device_id loclSelectDevice(int dev_idx, FILE *fout = NULL){
	cl_device_id device_selected = (cl_device_id)-1;

	cl_uint cnt_platforms, cnt_device_ids;
	cl_platform_id *platform_ids;
	OCL_SAFE_CALL( clGetPlatformIDs(0, NULL, &cnt_platforms) );

	platform_ids = (cl_platform_id*)alloca(sizeof(cl_platform_id)*cnt_platforms);
	OCL_SAFE_CALL( clGetPlatformIDs(cnt_platforms, platform_ids, NULL) );

	int cur_dev_idx = 1;
	for(int i=0; i<(int)cnt_platforms; i++){
		OCL_SAFE_CALL( clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, 0, NULL, &cnt_device_ids) );

		cl_device_id *device_ids = (cl_device_id*)alloca(sizeof(cl_device_id)*cnt_device_ids);
		OCL_SAFE_CALL( clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, cnt_device_ids, device_ids, NULL) );

		for(int d=0; d<(int)cnt_device_ids; d++){
			if( dev_idx==cur_dev_idx ){
				device_selected = device_ids[d];
				if( fout ){
					char dev_name[256];
					OCL_SAFE_CALL( clGetDeviceInfo(device_ids[d], CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL) );
					fprintf(fout, "Selected device: %s\n", dev_name);
				}
			}
			cur_dev_idx++;
		}
	}
	return device_selected;
}


inline cl_device_id loclGetArgSelectedDevice(int argc, char* argv[]){
	printf("\nInitializing OpenCL\n");

	int preferred_device = -1;
//#ifdef __LOCL_DEVICE_NO
//	preferred_device = __LOCL_DEVICE_NO;
//#else
	if( argc<2 ){
		loclPrintAvailableDevices();
		exit(EXIT_FAILURE);
	} else
		preferred_device = atoi(argv[1]);
//#endif

	cl_device_id device_selected = loclSelectDevice(preferred_device, stdout);

	if( device_selected==(cl_device_id)-1 ){
		if( preferred_device==-1 )
			fprintf(stderr, "No OpenCL device selected\n");
		else
			fprintf(stderr, "OpenCL with index %d does not exist\n", preferred_device);
		exit(EXIT_FAILURE);
	}
	return device_selected;
}

inline cl_program loclBuildProgram(cl_context context, cl_device_id device, const char* src, const char *options, bool verbose = true){
	const char *all_sources[1] = {src};
	cl_int errno;
	if( verbose )
		printf("Creating program...\t");
	cl_program program = clCreateProgramWithSource(context, 1, all_sources, NULL, &errno);
	OCL_SAFE_CALL(errno);
	if( verbose )
		printf("Ok\n");

	if( verbose )
		printf("Building program...\t");
	errno = clBuildProgram(program, 1, &device, options, NULL, NULL);
	if( verbose )
		printf("Ok\n");

	if( errno!=CL_SUCCESS ){
		char log[10000];
		OCL_SAFE_CALL( clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL) );
		OCL_SAFE_CALL( clReleaseProgram(program) );
		puts(log);
		exit(EXIT_FAILURE);
	}
	return program;
}

inline double loclGetEventExecTimeAndRelease(cl_event ev){
	cl_ulong ev_t_start, ev_t_finish;
	OCL_SAFE_CALL( clWaitForEvents(1, &ev) );
	OCL_SAFE_CALL( clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &ev_t_start, NULL) );
	OCL_SAFE_CALL( clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &ev_t_finish, NULL) );
	double time = (ev_t_finish-ev_t_start)/1000000000.0;
	OCL_SAFE_CALL( clReleaseEvent( ev ) );
	return time;
}

inline double loclGetEventExecTimeAndRelease(cl_event evfirst, cl_event evlast){
	cl_ulong ev_t_start, ev_t_finish;
//	cl_event event_list[2] = {evfirst, evlast};
	OCL_SAFE_CALL( clWaitForEvents(1, /*&evfirst*//*event_list*/&evlast) );
	OCL_SAFE_CALL( clGetEventProfilingInfo(evfirst, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &ev_t_start, NULL) );
	OCL_SAFE_CALL( clGetEventProfilingInfo(evlast, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &ev_t_finish, NULL) );
	double time = (ev_t_finish-ev_t_start)/1000000000.0;
	OCL_SAFE_CALL( clReleaseEvent( evfirst ) );
	OCL_SAFE_CALL( clReleaseEvent( evlast ) );
	return time;
}


#endif
