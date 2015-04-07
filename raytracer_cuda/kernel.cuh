#ifndef KERNEL_CUH
#define KERNEL_CUH

#include "job.cuh"

__global__ void ray_kernel(job_t job, int depth, scene_t *scene);

#endif
