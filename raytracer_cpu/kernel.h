#ifndef KERNEL_CUH
#define KERNEL_CUH

#include "job.h"
#include "scene.h"

extern int DEPTH_MAX;

void main_loop(job_t host_job, scene_t *scene);

extern float const_mem[15 * 1024];

#endif
