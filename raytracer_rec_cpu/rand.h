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

typedef struct
{
	uint32_t a, b, c, d;
} rand_init_t;

void init_fast_random(fast_random_t &fr, int uniq_id, int dest_id, rand_init_t init);

uint32_t rand_full(fast_random_t &fr);

float rand_f(fast_random_t &fr);

uint32_t rand256(fast_random_t &fr);

void init_rand(rand_init_t &rand_init);

#endif
