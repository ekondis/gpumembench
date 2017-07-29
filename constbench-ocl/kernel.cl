#define CMD_HELPER(FUNC, NAME) FUNC ## _ ## NAME
#define CMD(FUNC, NAME) CMD_HELPER(FUNC, NAME)

int reduce_int(int v)    { return v; }
int reduce_int2(int2 v)  { return v.x+v.y; }
int reduce_int4(int4 v)  { return v.x+v.y+v.z+v.w; }
int reduce_int8(int8 v)  { return v.s0+v.s1+v.s2+v.s3+v.s4+v.s5+v.s6+v.s7; }
int reduce_int16(int16 v){ return v.s0+v.s1+v.s2+v.s3+v.s4+v.s5+v.s6+v.s7+v.s8+v.s9+v.sA+v.sB+v.sC+v.sD+v.sE+v.sF; }

int init_val_int(int i){ return i; }
int2 init_val_int2(int i){ return (int2)(i, i); }
int4 init_val_int4(int i){ return (int4)(i, i, i, i); }

int first_element_int(int v)    { return v; }
int first_element_int2(int2 v)  { return v.x; }
int first_element_int4(int4 v)  { return v.x; }

__kernel void CMD(krn_benchmark_constant, __CUSTOM_TYPE__)(__global int *output, __constant __CUSTOM_TYPE__ *constant_data){
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
	if( globaltid==0 /*wid==0*/ )
		*output = res;
}