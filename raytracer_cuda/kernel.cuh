#ifndef KERNEL_CUH
#define KERNEL_CUH

#include "job.cuh"
#include "scene.cuh"

#define DEPTH_MAX 4

void main_loop(job_t host_job, scene_t *scene);

extern float __constant__ const_mem[15 * 1024];

#endif
