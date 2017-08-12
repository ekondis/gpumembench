#define CMD_HELPER(FUNC, NAME) FUNC ## _ ## NAME
#define CMD(FUNC, NAME) CMD_HELPER(FUNC, NAME)

int reduce_int(int v)    { return v; }
int reduce_int2(int2 v)  { return v.x+v.y; }
int reduce_int4(int4 v)  { return v.x+v.y+v.z+v.w; }
//int reduce_int8(int8 v)  { return v.s0+v.s1+v.s2+v.s3+v.s4+v.s5+v.s6+v.s7; }
//int reduce_int16(int16 v){ return v.s0+v.s1+v.s2+v.s3+v.s4+v.s5+v.s6+v.s7+v.s8+v.s9+v.sA+v.sB+v.sC+v.sD+v.sE+v.sF; }

int init_val_int(int i){ return i; }
int2 init_val_int2(int i){ return (int2)(i, i); }
int4 init_val_int4(int i){ return (int4)(i, i, i, i); }

int first_element_int(int v)    { return v; }
int first_element_int2(int2 v)  { return v.x; }
int first_element_int4(int4 v)  { return v.x; }

int assign_element_int(int4 v)  { return v.x; }
int2 assign_element_int2(int4 v)  { return (int2)(v.x, v.y); }
int4 assign_element_int4(int4 v)  { return (int4)(v.x, v.y, v.z, v.w); }

unsigned int displacement_int(int v, unsigned int offset)   { return v + offset; }
unsigned int displacement_int2(int2 v, unsigned int offset) { return v.x + v.y + offset; }
unsigned int displacement_int4(int4 v, unsigned int offset) { return v.x + v.y + v.z + v.w + offset; }

/*int load_int(const int *d, unsigned int offset)   { return d[offset]; }
int2 load_int2(const int2 *d, unsigned int offset) { return v.x + v.y + v2; }
int4 load_int4(const int4 *d, unsigned int offset) { return v.x + v.y + v.z + v.w + v2; }*/

/*__kernel void CMD(krn_benchmark_constant, __CUSTOM_TYPE__)(__global int *output, __constant __CUSTOM_TYPE__ *constant_data){
	int tid = get_local_id(0);
	int wid = get_group_id(0);
	int globaltid = get_global_id(0);
	int blockDim = get_local_size(0);
	__CUSTOM_TYPE__ sum = (__CUSTOM_TYPE__)(0);
	int offset = CMD(first_element, __CUSTOM_TYPE__)(constant_data[tid])-1;
	__constant __CUSTOM_TYPE__ *constant_data_base = &constant_data[offset];
	// Force 4 wide strides in order to avoid automatic merging of multiple accesses to 128bit accesses
	for(int i=0; i<4; i++){
#pragma unroll 128
		for(int j=0; j<VECTOR_SIZE/(sizeof(__CUSTOM_TYPE__)/sizeof(int)); j+=4){
			sum += constant_data_base[j+i];
		}
	}
	int res = CMD(reduce, __CUSTOM_TYPE__)(sum);
	if( globaltid==0 )
		*output = res;
}*/



__kernel void fill_buffer(unsigned int len, unsigned int v, __global unsigned int *b){
	int gid = get_global_id(0);
	while( gid<len ){
		b[gid] = v;
		gid += get_global_size(0);
	}
}

__kernel  __attribute__((reqd_work_group_size(BLOCK_SIZE, 1, 1)))
void benchmark_func(__global __CUSTOM_TYPE__ *g_data
#ifdef TEX_LOADS
	, __read_only image2d_t img_data
#endif
	){
	const int grid_data_width = STEPWIDTH*get_global_size(0);
	const bool set_readonly =
#ifdef READONLY
		true;
#else
		false;
#endif

	// Thread block-wise striding
	int index = STEPWIDTH*get_group_id(0)*BLOCK_SIZE + get_local_id(0);
	index = INDEX_CLAMPING==0 ? index : index % INDEX_CLAMPING;
	const int stride = BLOCK_SIZE;

	unsigned int offset = index;
	__CUSTOM_TYPE__ temp = CMD(init_val, __CUSTOM_TYPE__)(0);
	for(int j=0; j<TOTAL_ITERATIONS; j+=UNROLL_ITERATIONS){
		// Pretend updating of offset in order to force repetitive loads
		offset = CMD(displacement, __CUSTOM_TYPE__)(temp, offset);
#ifndef TEX_LOADS
		union {
#ifdef VOL_LOADS
		volatile
#endif
		__global const __CUSTOM_TYPE__ *ptr;
			size_t i;
		} g_data_load_ptr = { g_data+offset };
#endif
		__global __CUSTOM_TYPE__ * const g_data_store_ptr = g_data+offset+grid_data_width;
#pragma unroll
		for(int i=0; i<UNROLL_ITERATIONS; i++){
			const unsigned int iteration_offset = (set_readonly ? i : i >> 1) % STEPWIDTH;
			if( set_readonly || (i % 2 == 0) ){
				
#ifdef TEX_LOADS
				const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
				const size_t flat_offset = offset+iteration_offset*stride;
				const int2 image_offset = {flat_offset % IMAGE_ROW_LEN, flat_offset / IMAGE_ROW_LEN};
				const __CUSTOM_TYPE__ v = CMD(assign_element, __CUSTOM_TYPE__)( read_imagei(img_data, sampler, image_offset) );
				//const __CUSTOM_TYPE__ v = CMD(load, __CUSTOM_TYPE__)(g_data, offset+iteration_offset*stride);
#else
				const __CUSTOM_TYPE__ v = g_data_load_ptr.ptr[iteration_offset*stride];
#endif
				if( set_readonly ){
					// Pretend update to index in order to force reloads
#ifdef TEX_LOADS
					offset ^=
#else
					g_data_load_ptr.i ^=
#endif
						CMD(reduce, __CUSTOM_TYPE__)(v);
				}
				temp = v;
			} else
				g_data_store_ptr[iteration_offset*stride] = temp;
		}
	}
	offset = CMD(displacement, __CUSTOM_TYPE__)(temp, offset);
	if( offset != index ) // Does not occur
		*g_data = CMD(init_val, __CUSTOM_TYPE__)(offset);
}


/*

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
		T * const g_data_store_ptr = g_data+offset+grid_data_width;
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
*/

