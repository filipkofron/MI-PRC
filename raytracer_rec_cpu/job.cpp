#include "job.cuh"

#include "common.cuh"

int calc_jobs(int real_job_num)
{
  return (real_job_num / THREADS_PER_BLOCK) * THREADS_PER_BLOCK + ((real_job_num % THREADS_PER_BLOCK) ? THREADS_PER_BLOCK : 0);
}

job_t allocate_host_job(job_t job)
{
  job_t host_job = job;
  int jobs_num = calc_jobs(host_job.image_width * host_job.image_height);
  safeMalloc((void **)&host_job.image_dest, sizeof(float) * jobs_num * 3);
  safeMalloc((void **)&host_job.ray_pos, sizeof(float) * jobs_num * 3);
  safeMalloc((void **)&host_job.ray_dir, sizeof(float) * jobs_num * 3);

  return host_job;
}

void free_host_job(job_t *host_job)
{
  safeFree(host_job->image_dest);
  host_job->image_dest = NULL;
  safeFree(host_job->ray_pos);
  host_job->ray_pos = NULL;
  safeFree(host_job->ray_dir);
  host_job->ray_dir = NULL;
}

job_t allocate_device_job(job_t job)
{
  job_t dev_job = job;
  int jobs_num = calc_jobs(dev_job.image_width * dev_job.image_height);
  cudaSafeMalloc((void **)&dev_job.image_dest, sizeof(float) * jobs_num * 3);
  cudaSafeMalloc((void **)&dev_job.ray_pos, sizeof(float) * jobs_num * 3);
  cudaSafeMalloc((void **)&dev_job.ray_dir, sizeof(float) * jobs_num * 3);

  return dev_job;
}

void free_device_job(job_t *dev_job)
{
  cudaSafeFree(dev_job->image_dest);
  dev_job->image_dest = NULL;
  cudaSafeFree(dev_job->ray_pos);
  dev_job->ray_pos = NULL;
  cudaSafeFree(dev_job->ray_dir);
  dev_job->ray_dir = NULL;
}


void copy_job_to_dev(job_t *dev_dest, job_t *host_src)
{
  dev_dest->image_width = host_src->image_width;
  dev_dest->image_height = host_src->image_height;
  dev_dest->pass_count = host_src->pass_count;
  int hc = calc_jobs(host_src->image_width * host_src->image_height);
  cudaMemcpy(dev_dest->image_dest, host_src->image_dest, hc * sizeof(float) * 3, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_dest->ray_pos, host_src->ray_pos, hc * sizeof(float) * 3, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_dest->ray_dir, host_src->ray_dir, hc * sizeof(float) * 3, cudaMemcpyHostToDevice);
}

void copy_job_to_host(job_t *host_dest, job_t *dev_src)
{
  host_dest->image_width = dev_src->image_width;
  host_dest->image_height = dev_src->image_height;
  host_dest->pass_count = dev_src->pass_count;
  int hc = calc_jobs(dev_src->image_width * dev_src->image_height);
  cudaMemcpy(host_dest->image_dest, dev_src->image_dest, hc * sizeof(float) * 3, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_dest->ray_pos, dev_src->ray_pos, hc * sizeof(float) * 3, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_dest->ray_dir, dev_src->ray_dir, hc * sizeof(float) * 3, cudaMemcpyDeviceToHost);
}
