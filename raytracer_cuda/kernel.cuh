#ifndef KERNEL_CUH
#define KERNEL_CUH

#include "job.cuh"

__global__ void ray_kernel(job_t job, int depth, scene_t *scene);
void main_loop(job_t host_job, scene_t *scene);

#endif
