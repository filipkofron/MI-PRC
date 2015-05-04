#include "rand.h"
#include <cstdlib>

void init_fast_random(fast_random_t &fr, int uniq_id, int dest_id, rand_init_t init)
{
	int init2 = init.a * 17;
	int init3 = init.b * 59;
	int init4 = init.c * 23;
	fr.x = uniq_id * init.d + dest_id * init3;
	fr.y = uniq_id * init2 + dest_id * init2;
	fr.z = uniq_id * init3 + dest_id * init3;
	fr.w = uniq_id * init4 + dest_id * init.d;
}

uint32_t rand_full(fast_random_t &fr)
{
	uint32_t t = fr.x ^ (fr.x << 11);
	fr.x = fr.y;
	fr.y = fr.z;
	fr.z = fr.w;
	return fr.w = fr.w ^ (fr.w >> 19) ^ t ^ (t >> 8);
}

float rand_f(fast_random_t &fr)
{
	uint32_t r = rand_full(fr);
	return 100.0f / ((r & 0xFFFF) + 1);
}

uint32_t rand256(fast_random_t &fr)
{
	uint32_t t = fr.x ^ (fr.x << 11);
	fr.x = fr.y;
	fr.y = fr.z;
	fr.z = fr.w;
	return (fr.w = fr.w ^ (fr.w >> 19) ^ t ^ (t >> 8)) & 255;
}

void init_rand(rand_init_t &rand_init)
{
	rand_init.a = rand();
	rand_init.b = rand();
	rand_init.c = rand();
	rand_init.d = rand();
}
