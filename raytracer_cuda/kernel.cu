#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.cuh"
#include "bmp.cuh"
#include "scene.cuh"
#include "trace.cuh"
#include "job.cuh"

#include <cstdio>
#include <iostream>
#include <stack>
#include <assert.h>

__global__ void init_kernel(job_t job)
{
	int uniq_id = threadIdx.x + blockIdx.x * blockDim.x;

	int kernel_x = uniq_id % job.image_width;
	int kernel_y = uniq_id / job.image_width;

	float norm = (job.image_width > job.image_height ? job.image_width : job.image_height);

	init_vec3(&job.ray_pos[uniq_id * 3], (kernel_x - 0.5f * job.image_width) / norm, (kernel_y - 0.5f * job.image_height) / norm, -60.0f);

	float off_x = job.ray_pos[uniq_id * 3] * 0.001f;
	float off_y = job.ray_pos[uniq_id * 3 + 1] * 0.001f;

	init_vec3(&job.ray_dir[uniq_id * 3], off_x, off_y, 1.0f);
	normalize(&job.ray_dir[uniq_id * 3]);

	float *color_offset = &job.image_dest[uniq_id * 3];
	color_offset[0] = color_offset[1] = color_offset[2] = 0.0f;
}

__global__ void ray_kernel(job_t job, int depth, scene_t *scene)
{
	int uniq_id = threadIdx.x + blockIdx.x * blockDim.x;

	// All threads do work but some of the results will be discarded as they are only the padding in the arrays.

	trace_ray(
		&job.gather_arr[uniq_id],			// assign the result gather array
		&job.target_idx[uniq_id],			// assign the target index array
		&job.image_dest[uniq_id * 3],	// assign the result color array
		&job.ray_pos[uniq_id * 3],		// assign job ray position
		&job.ray_dir[uniq_id * 3],		// assign job ray direction
		depth,												// this shall stop the recursion
		scene);												// const scene
}

__global__ void forward_kernel(job_t old_job, job_t new_job)
{
	int uniq_id = threadIdx.x + blockIdx.x * blockDim.x;

	if(old_job.gather_arr[uniq_id])
	{
		int dest_id = old_job.target_idx[uniq_id];

		set_vec3(&new_job.ray_pos[dest_id * 3], &old_job.ray_pos[uniq_id * 3]);

		// TODO: distribute randomly, mabe both
		set_vec3(&new_job.ray_dir[dest_id * 3], &old_job.ray_dir[uniq_id * 3]);
	}
}

__global__ void pps_kernel(int *dest, int *src, int powerof2Minus1)
{
	int uniq_id = threadIdx.x + blockIdx.x * blockDim.x;

	if(uniq_id >= (powerof2Minus1 << 1))
		dest[uniq_id] = src[uniq_id - powerof2Minus1] + src[uniq_id];
	else
		dest[uniq_id] = src[uniq_id];

		// TODO: Optimize!!!
}

static void do_pps(int *arr, int size)
{
	int d_max = ceil_log2(size);
	int *temp = NULL;
	cudaSafeMalloc((void **) &temp, sizeof(int) * size);
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

	cudaSafeFree(temp);
}

static int ray_step(job_t dev_job, scene_t *scene, int depth)
{
	int size = calc_jobs(dev_job.image_width * dev_job.image_height);
	assert(size > 0);

	if(depth == 0)
	{
		// we have to set the first pos to be that of screen
		std::cout << "[MainLoop] >> Running init_kernel to setup job seed." << std::endl;
		init_kernel<<< BLOCKS_PER_JOB(size), size % THREADS_PER_BLOCK >>>(dev_job);
	}

	ray_kernel<<< BLOCKS_PER_JOB(size), size % THREADS_PER_BLOCK >>>(dev_job, depth, scene);
	int next_size = 0;
	do_pps(dev_job.target_idx, size);
	cudaMemcpy(&next_size, &dev_job.target_idx[size - 1], sizeof(int), cudaMemcpyDeviceToHost);
	return next_size;
}

void main_loop(job_t host_job, scene_t *scene)
{
	std::cout << "[MainLoop] >> Begin with image size: " << host_job.image_width << "x" << host_job.image_height << " for " << host_job.pass_count << "passes" << std::endl;
	int depth = 0;

	std::stack<job_t> jobs;

	job_t temp_job = host_job;
	job_t curr_job = allocate_device_job(temp_job);

	//buildup
	while(depth < 4)
	{
		std::cout << "[MainLoop] >> Stage " << depth << " start." << std::endl;
		int next_size = ray_step(curr_job, scene, depth);
		std::cout << "[MainLoop] >> Ray resulted in " << next_size << " following jobs." << std::endl;
		if(next_size)
		{
			int size = calc_jobs(next_size);
			temp_job.image_width = next_size % THREADS_PER_BLOCK;
			temp_job.image_height = next_size / THREADS_PER_BLOCK;
			temp_job = allocate_device_job(temp_job);
			forward_kernel<<< BLOCKS_PER_JOB(size), size % THREADS_PER_BLOCK >>>(temp_job, curr_job);
			jobs.push(curr_job);
			curr_job = temp_job;
		}
		else
		{
			break;
		}
		depth++;
	}

	std::cout << "[MainLoop] >> TODO: Unwind." << std::endl;
	//unwind
	// TODO: unwind & merge, free
}
