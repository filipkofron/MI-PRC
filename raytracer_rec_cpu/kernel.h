#ifndef KERNEL_CUH
#define KERNEL_CUH

#include "job.h"
#include "scene.h"

/*
#define CPU_KERNEL(kernel, a, b, ...) \
  do {    \
    for(int bl_x = 0; bl_x < a; bl_x++) \
      for(int th_x = 0; th_x < b; th_x++) \
        kernel(th_x, bl_x, b, __VA_ARGS__);\
  } while(false);
*/

extern int DEPTH_MAX;

void main_loop(job_t host_job, scene_t *scene);

extern float *const_mem;

#endif
