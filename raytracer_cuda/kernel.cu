#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.cuh"
#include "bmp.cuh"
#include "scene.cuh"
#include "trace.cuh"
#include "job.cuh"
#include "kernel.cuh"
#include "rand.cuh"

#include <cstdio>
#include <iostream>
#include <stack>
#include <assert.h>

float __constant__ const_mem[15 * 1024];

__global__ void init_kernel(job_t job)
{
	int uniq_id = threadIdx.x + blockIdx.x * blockDim.x;

	int kernel_x = uniq_id % job.image_width;
	int kernel_y = uniq_id / job.image_width;

	float norm = (job.image_width > job.image_height ? job.image_width : job.image_height);
	float inv_norm = 1.0f / norm;
	init_vec3(&job.ray_pos[uniq_id * 3], (kernel_x - 0.5f * job.image_width) * inv_norm, (kernel_y - 0.5f * job.image_height) * inv_norm, -60.0f);

	float off_x = job.ray_pos[uniq_id * 3] * 0.1f;
	float off_y = job.ray_pos[uniq_id * 3 + 1] * 0.1f;

	init_vec3(&job.ray_dir[uniq_id * 3], off_x, off_y, 1.0f);
	normalize(&job.ray_dir[uniq_id * 3]);

	float *color_offset = &job.image_dest[uniq_id * 3];
	init_vec3(color_offset, 0.0f, 0.0f, 0.0f);
}

__global__ void ray_kernel(job_t job, int depth, scene_t scene)
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
		&scene);												// const scene
		clamp(&job.image_dest[uniq_id * 3]);
}

__global__ void forward_kernel(job_t old_job, job_t new_job)
{
	int uniq_id = threadIdx.x + blockIdx.x * blockDim.x;
	int dest_id = old_job.target_idx[uniq_id] - old_job.gather_arr[uniq_id];
	int max_id = dest_id + old_job.gather_arr[uniq_id];

	//printf("uniq: %i, dest_id: %i, max_id: %i\n", uniq_id, dest_id, max_id);

	// remember that the PPS will start with 1
	for(int dest_idx = dest_id; dest_idx < max_id; dest_idx++)
	{
		set_vec3(&new_job.ray_pos[dest_idx * 3], &old_job.ray_pos[uniq_id * 3]);

		set_vec3(&new_job.ray_dir[dest_idx * 3], &old_job.ray_dir[uniq_id * 3]);
	}
}

__global__ void rand_kernel(job_t old_job, job_t new_job, int rand_init)
{
	int uniq_id = threadIdx.x + blockIdx.x * blockDim.x;
	int dest_id = old_job.target_idx[uniq_id] - old_job.gather_arr[uniq_id];
	int max_id = dest_id + old_job.gather_arr[uniq_id];
	float rv[3];

	//printf("uniq: %i, dest_id: %i, max_id: %i\n", uniq_id, dest_id, max_id);
	fast_random_t frand;
	init_fast_random (frand, uniq_id, dest_id, rand_init);

	// remember that the PPS will start with 1
	for(int dest_idx = dest_id; dest_idx < max_id; dest_idx++)
	{
		init_vec3(rv, rand_f(frand), rand_f(frand), rand_f(frand));
		//printf("rv0: %f, rv1: %f, rv2: %f\n", rv[0], rv[1], rv[2]);
		add(&new_job.ray_dir[dest_idx * 3], rv);
	}
}

__global__ void backward_kernel(job_t target, job_t prev)
{
	int uniq_id = threadIdx.x + blockIdx.x * blockDim.x;
	int gath = target.gather_arr[uniq_id];
	int dest_id = target.target_idx[uniq_id] - gath;
	int max_id = dest_id + gath;

	float *res_addr = &target.image_dest[uniq_id * 3];
	float temp_res[3];
	set_vec3(temp_res, res_addr);
	float part = 1.0f / gath;

	float temp[3];
	for(int dest_idx = dest_id; dest_idx < max_id; dest_idx++)
	{
		set_vec3(temp, &prev.image_dest[dest_idx * 3]);
		mul(temp, part);
		add(temp_res, temp);
	}
	clamp(temp_res);
	set_vec3(res_addr, temp_res);
}

__global__ void pps_kernel(int *dest, int *src, int powerof2)
{
	int uniq_id = threadIdx.x + blockIdx.x * blockDim.x;

	if(uniq_id >= powerof2)
		dest[uniq_id] = src[uniq_id - powerof2] + src[uniq_id];
	else
		dest[uniq_id] = src[uniq_id];

		// TODO: Optimize!!!
}

static void do_pps(int *arr, int size)
{
	std::cout << "[PPS] >> For size: " << size * 4 << " B" << std::endl;
	int d_max = ceil_log2(size);
	int *temp = NULL;
	cudaSafeMalloc((void **) &temp, sizeof(int) * size);
	int *orig_temp = temp;
	for(int d = 0; d <= d_max; d++)
	{
		pps_kernel<<< BLOCKS_PER_JOB(size), THREADS_PER_BLOCK >>>(temp, arr, pow2(d));
		cudaDeviceSynchronize();
		cudaCheckErrors("pps_kernel fail");
		int *swap = temp;
		temp = arr;
		arr = swap;
	}
	if((~d_max) & 1)
	{
		cudaMemcpy(arr, temp, size * sizeof(int), cudaMemcpyDeviceToDevice);
		cudaCheckErrors("MemCPY PPS fail");
	}

	cudaSafeFree(orig_temp);
}

static int ray_step(job_t dev_job, scene_t *scene, int depth)
{
	int size = calc_jobs(dev_job.image_width * dev_job.image_height);
	assert(size > 0);

	if(depth == 0)
	{
		// we have to set the first pos to be that of screen
		std::cout << "[MainLoop] >> Running init_kernel to setup job seed." << std::endl;
		init_kernel<<< BLOCKS_PER_JOB(size), THREADS_PER_BLOCK >>>(dev_job);
		cudaDeviceSynchronize();
		cudaCheckErrors("init_kernel fail");
	}

	std::cout << "[MainLoop] >> Ray tracing kernel .. " << std::endl;
	ray_kernel<<< BLOCKS_PER_JOB(size), THREADS_PER_BLOCK >>>(dev_job, depth, *scene);
	cudaDeviceSynchronize();
	cudaCheckErrors("ray_kernel fail");
	int next_size = 0;
	do_pps(dev_job.target_idx, size);
	cudaMemcpy(&next_size, &dev_job.target_idx[size - 1], sizeof(int), cudaMemcpyDeviceToHost);
	cudaCheckErrors("MemCPY the PPS max fail");
	return next_size;
}

void main_loop(job_t host_job, scene_t *scene)
{
	std::cout << "[MainLoop] >> Begin with image size: " << host_job.image_width << "x" << host_job.image_height << " for " << host_job.pass_count << " passes" << std::endl;
	int depth = 0;

	std::stack<job_t> jobs;

	job_t temp_job = host_job;
	job_t curr_job = allocate_device_job(temp_job);
	jobs.push(curr_job);

	//buildup
	while(depth < DEPTH_MAX)
	{
		std::cout << "[MainLoop] >> Stage " << depth << " start." << std::endl;
		int next_size = ray_step(curr_job, scene, depth);
		std::cout << "[MainLoop] >> Ray resulted in " << next_size << " following jobs." << std::endl;
		if(next_size && (depth + 1) < DEPTH_MAX)
		{
			int size = calc_jobs(next_size);
			temp_job.image_width = THREADS_PER_BLOCK;
			temp_job.image_height = next_size / THREADS_PER_BLOCK + (next_size % THREADS_PER_BLOCK ? 1 : 0);
			temp_job = allocate_device_job(temp_job);
			cudaDeviceSynchronize();
			cudaCheckErrors("pre-forward_kernel sync fail");
			int old_jobs_size = calc_jobs(curr_job.image_width * curr_job.image_height);
			std::cout << "[MainLoop oldJob] size: " << old_jobs_size << std::endl;
			std::cout << "[MainLoop newJob] size: " << calc_jobs(temp_job.image_width * temp_job.image_height) << std::endl;

			int rand_init = rand();

			forward_kernel<<< BLOCKS_PER_JOB(old_jobs_size), THREADS_PER_BLOCK >>>(curr_job, temp_job);
			cudaCheckErrors("forward_kernel fail");
			cudaDeviceSynchronize();
			cudaCheckErrors("forward_kernel sync fail");

			rand_kernel<<< BLOCKS_PER_JOB(old_jobs_size), THREADS_PER_BLOCK >>>(curr_job, temp_job, rand_init);
			cudaCheckErrors("rand_kernel fail");
			cudaDeviceSynchronize();
			cudaCheckErrors("rand_kernel sync fail");

			curr_job = temp_job;
			jobs.push(curr_job);
		}
		else
		{
			break;
		}
		depth++;
	}

	std::cout << "[MainLoop] >> TODO: Unwind." << std::endl;

	int back_i = 0;
	job_t back_job;
	while(jobs.size() > 0)
	{
		job_t old_job = jobs.top();
		jobs.pop();

		if(back_i)
		{
			int size = calc_jobs(old_job.image_width * old_job.image_height);
			std::cout << "backward: " << back_i << std::endl;
			backward_kernel<<< BLOCKS_PER_JOB(size), THREADS_PER_BLOCK >>>(old_job, back_job);
		}

		if(jobs.size() == 0)
		{
			copy_job_to_host(&host_job, &old_job);
		}
		if(back_i)
		{
			free_device_job(&back_job);
		}
		back_job = old_job;

		back_i++;
	}
	if(back_i)
		free_device_job(&back_job);
}
