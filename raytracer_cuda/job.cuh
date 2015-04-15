#ifndef JOB_CUH
#define JOB_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

typedef struct
{
  int image_width;
  int image_height;
  int pass_count;

  int *gather_arr;
  float *image_dest;
  float *ray_pos;
  float *ray_dir;
  int *target_idx;
} job_t;

#define THREADS_PER_BLOCK 32
//#define THREADS_PER_BLOCK 64
#define BLOCKS_PER_JOB(size) ((size) / (THREADS_PER_BLOCK))

int calc_jobs(int real_job_num);

job_t allocate_host_job(job_t job);
void free_host_job(job_t *host_job);

void copy_job_to_dev(job_t *dev_dest, job_t *host_src);
void copy_job_to_host(job_t *host_dest, job_t *dev_src);

job_t allocate_device_job(job_t job);
void free_device_job(job_t *dev_job);

#endif
