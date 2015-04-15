#ifndef RAND_HPP
#define RAND_HPP

#include <stdint.h>

#define INIT_RAND(arr, size) \
	for(int _arr_rand_init = 0; _arr_rand_init < size; _arr_rand_init++) \
	{ \
		arr[_arr_rand_init] = rand();\
	}

typedef struct
{
	uint32_t x, y, z, w;
} fast_random_t;

__device__ void init_fast_random(fast_random_t &fr, int uniq_id, int dest_id, int init)
{
	int init2 = init * 17;
	int init3 = init2 * 59;
	int init4 = init3 * 23;
	fr.x = uniq_id * init + dest_id * init;
	fr.y = uniq_id * init2 + dest_id * init2;
	fr.z = uniq_id * init3 + dest_id * init3;
	fr.w = uniq_id * init4 + dest_id * init4;
}

__device__ uint32_t rand_full(fast_random_t &fr)
{
	uint32_t t = fr.x ^ (fr.x << 11);
	fr.x = fr.y;
	fr.y = fr.z;
	fr.z = fr.w;
	return fr.w = fr.w ^ (fr.w >> 19) ^ t ^ (t >> 8);
}

__device__ float rand_f(fast_random_t &fr)
{
	uint32_t r = rand_full(fr);
	return 100.0f / ((r & 0xFFFF) + 1);
}

__device__ uint32_t rand256(fast_random_t &fr)
{
	uint32_t t = fr.x ^ (fr.x << 11);
	fr.x = fr.y;
	fr.y = fr.z;
	fr.z = fr.w;
	return (fr.w = fr.w ^ (fr.w >> 19) ^ t ^ (t >> 8)) & 255;
}

#endif
