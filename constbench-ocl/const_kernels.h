/**
 * const_kernels.h: This file is part of the gpumembench micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#pragma once

#define VECTOR_SIZE (1*1024)

extern "C" void constbenchGPU(cl_device_id, int*, size_t);
