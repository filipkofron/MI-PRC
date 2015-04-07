#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.cuh"
#include "bmp.cuh"
#include "scene.cuh"
#include "trace.cuh"
#include "job.cuh"

#include <cstdio>
#include <iostream>

__global__ void ray_kernel(job_t job, int depth, scene_t *scene)
{
	int uniq_id = threadIdx.x + blockIdx.x * blockDim.x;

	// All threads do work but some of the results will be discarded as they are only the padding in the arrays.

	int kernel_x = uniq_id % blockDim.x;
	int kernel_y = uniq_id / blockDim.x;

	trace_ray(
		&job.image_dest[uniq_id * 3],	// assign the result color array
		&job.ray_pos[uniq_id * 3],		// assign job ray position
		&job.ray_dir[uniq_id * 3],		// assign job ray direction
		depth,												// this shall stop the recursion
		scene);												// const scene
}

__global__ void pps_kernel(int *dest, int *src, int powerof2Minus1)
{
	int uniq_id = threadIdx.x + blockIdx.x * blockDim.x;
	dest[uniq_id] = src[uniq_id - powerof2Minus1] + src[uniq_id];
}

static void do_pps(int *arr, int size)
{
	int d_max = ceil_log2(size);
	int *temp = NULL;
	cudaSafeMalloc(&temp, sizeof(int) * size);
	for(int d = 1; d <= d_max; d++)
	{
		pps_kernel<<< BLOCKS_PER_JOB(size), THREADS_PER_BLOCK >>>(temp, arr, pow2(d - 1));
		int *swap = temp;
		temp = arr;
		arr = swap;
	}
	if(d_max & 1)
	{
		cudaMemcpy(arr, temp, size * sizeof(int), cudaMemcpyDeviceToDevice);
	}
}

static int step(job_t dev_job, scene_t *scene, int depth)
{
	int size = calc_jobs(dev_job.image_width * dev_job.image_height)
	assert(size > 0);
	ray_kernel(dev_job, depth, scene);
	int next_size = 0;
	do_pps(dev_job.target_idx, size);
	cudaMemcpy(&dest_size, &dev_job.target_idx[size - 1], sizeof(int), cudaMemcpyDeviceToHost);
	return next_size;
}

void main_loop(job_t host_job, scene_t *scene)
{

}
