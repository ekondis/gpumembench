/**
 * cache_kernels.h: This file is part of the gpumembench micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#pragma once

#define VECTOR_SIZE (32*1024*1024)

enum bench_type{ btAllCache, btL2Cache, btTexture };

extern "C" void cachebenchGPU(cl_device_id, double*, size_t, bench_type, bool);
