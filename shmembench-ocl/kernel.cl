#define TOTAL_ITERATIONS (1024)
#define DATA_TYPE int

#define CMD_HELPER(FUNC, NAME) FUNC ## _ ## NAME
#define CMD(FUNC, NAME) CMD_HELPER(FUNC, NAME)

// shared memory swap operation (2 floats read + 2 floats write)
void shmem_swap_int(__local int *v1, __local int *v2){
	int tmp;
	tmp = *v2;
	*v2 = *v1;
	*v1 = tmp;
}

void shmem_swap_int2(__local int2 *v1, __local int2 *v2){
	int2 tmp;
	tmp = *v2;
	*v2 = *v1;
	*v1 = tmp;
}

void shmem_swap_int4(__local int4 *v1, __local int4 *v2){
	int4 tmp;
	tmp = *v2;
	*v2 = *v1;
	*v1 = tmp;
}

int reduce_int(int v)    { return v; }
int reduce_int2(int2 v)  { return v.x+v.y; }
int reduce_int4(int4 v)  { return v.x+v.y+v.z+v.w; }
int reduce_int8(int8 v)  { return v.s0+v.s1+v.s2+v.s3+v.s4+v.s5+v.s6+v.s7; }
int reduce_int16(int16 v){ return v.s0+v.s1+v.s2+v.s3+v.s4+v.s5+v.s6+v.s7+v.s8+v.s9+v.sA+v.sB+v.sC+v.sD+v.sE+v.sF; }

int reduce_vector_int(int v1, int v2, int v3, int v4, int v5, int v6){
	return v1 + v2 + v3 + v4 + v5 + v6;
}

int2 reduce_vector_int2(int2 v1, int2 v2, int2 v3, int2 v4, int2 v5, int2 v6){
	return (int2)(v1.x + v2.x + v3.x + v4.x + v5.x + v6.x, v1.y + v2.y + v3.y + v4.y + v5.y + v6.y);
}

int4 reduce_vector_int4(int4 v1, int4 v2, int4 v3, int4 v4, int4 v5, int4 v6){
	return (int4)(v1.x + v2.x + v3.x + v4.x + v5.x + v6.x, v1.y + v2.y + v3.y + v4.y + v5.y + v6.y, v1.z + v2.z + v3.z + v4.z + v5.z + v6.z, v1.w + v2.w + v3.w + v4.w + v5.w + v6.w);
}

int init_val_int(int i){
	return i;
}

int2 init_val_int2(int i){
	return (int2)(i, i+11);
}

int4 init_val_int4(int i){
	return (int4)(i, i+11, i+19, i+23);
}

void set_vector_int(__local int *target, int offset, int v){
	target[offset] = v;
}

void set_vector_int2(__local int2 *target, int offset, int2 v){
	target[offset].x = v.x;
	target[offset].y = v.y;
}

void set_vector_int4(__local int4 *target, int offset, int4 v){
	target[offset].x = v.x;
	target[offset].y = v.y;
	target[offset].z = v.z;
	target[offset].w = v.w;
}

#define CUSTOM_COMMAND(name, suffix) name ## suffix

__kernel void CMD(krn_localmem_juggling, __CUSTOM_TYPE__)(__global int *g_data, __local __CUSTOM_TYPE__ *shm_buffer){
	int tid = get_local_id(0);
	int globaltid = get_global_id(0);
	int blockDim = get_local_size(0);
	CMD(set_vector, __CUSTOM_TYPE__)(shm_buffer, tid+0*blockDim, CMD(init_val, __CUSTOM_TYPE__)(tid));
	CMD(set_vector, __CUSTOM_TYPE__)(shm_buffer, tid+1*blockDim, CMD(init_val, __CUSTOM_TYPE__)(tid+1));
	CMD(set_vector, __CUSTOM_TYPE__)(shm_buffer, tid+2*blockDim, CMD(init_val, __CUSTOM_TYPE__)(tid+3));
	CMD(set_vector, __CUSTOM_TYPE__)(shm_buffer, tid+3*blockDim, CMD(init_val, __CUSTOM_TYPE__)(tid+7));
	CMD(set_vector, __CUSTOM_TYPE__)(shm_buffer, tid+4*blockDim, CMD(init_val, __CUSTOM_TYPE__)(tid+13));
	CMD(set_vector, __CUSTOM_TYPE__)(shm_buffer, tid+5*blockDim, CMD(init_val, __CUSTOM_TYPE__)(tid+17));
	mem_fence( CLK_LOCAL_MEM_FENCE );
#pragma unroll 32
	for(int j=0; j<TOTAL_ITERATIONS; j++){
		CMD(shmem_swap, __CUSTOM_TYPE__)(shm_buffer+tid+0*blockDim, shm_buffer+tid+1*blockDim);
		CMD(shmem_swap, __CUSTOM_TYPE__)(shm_buffer+tid+2*blockDim, shm_buffer+tid+3*blockDim);
		CMD(shmem_swap, __CUSTOM_TYPE__)(shm_buffer+tid+4*blockDim, shm_buffer+tid+5*blockDim);
		mem_fence( CLK_LOCAL_MEM_FENCE );
		CMD(shmem_swap, __CUSTOM_TYPE__)(shm_buffer+tid+1*blockDim, shm_buffer+tid+2*blockDim);
		CMD(shmem_swap, __CUSTOM_TYPE__)(shm_buffer+tid+3*blockDim, shm_buffer+tid+4*blockDim);
		mem_fence( CLK_LOCAL_MEM_FENCE );
	}
	g_data[globaltid] = CMD(reduce, __CUSTOM_TYPE__)( CMD(reduce_vector, __CUSTOM_TYPE__)(shm_buffer[tid+0*blockDim], shm_buffer[tid+1*blockDim], shm_buffer[tid+2*blockDim], shm_buffer[tid+3*blockDim], shm_buffer[tid+4*blockDim], shm_buffer[tid+5*blockDim]) );
}
