#include "common.h"
#include "bmp.h"
#include "scene.h"
#include "trace.h"
#include "job.h"
#include "kernel.h"
#include "rand.h"

#include <cstdio>
#include <iostream>
#include <stack>
#include <assert.h>
#include <thread>
#include <cstring>

int DEPTH_MAX = 1;
#define MIN(a, b) ((a) < (b) ? (a) : (b))
float const_mem[15 * 1024];

void init_kernel(int threadIdx_x, int blockIdx_x, int blockDim_x, job_t job, rand_init_t init)
{
	int uniq_id = threadIdx_x + blockIdx_x * blockDim_x;

	int kernel_x = uniq_id % job.image_width;
	int kernel_y = uniq_id / job.image_width;

	fast_random_t frand;
	init_fast_random (frand, kernel_x, kernel_y, init);

	float norm = (job.image_width > job.image_height ? job.image_width : job.image_height);
	float inv_norm = 1.0f / norm;
	init_vec3(&job.ray_pos[uniq_id * 3], (kernel_x - 0.5f * job.image_width) * inv_norm, (kernel_y - 0.5f * job.image_height) * inv_norm, -60.0f);

	float off_x = job.ray_pos[uniq_id * 3] * 0.1f + rand_f(frand) * 0.02;
	float off_y = job.ray_pos[uniq_id * 3 + 1] * 0.1f + rand_f(frand) * 0.02;

	init_vec3(&job.ray_dir[uniq_id * 3], off_x, off_y, 1.0f);
	normalize(&job.ray_dir[uniq_id * 3]);

	float *color_offset = &job.image_dest[uniq_id * 3];
	init_vec3(color_offset, 0.0f, 0.0f, 0.0f);
}

void init_kernel_th(int th_max, int blockIdx_x, int blockDim_x, job_t job, rand_init_t init)
{
	for(int th = 0; th < th_max; th++)
		for(int th_x = 0; th_x < blockDim_x; th_x++)
			init_kernel(th_x, blockIdx_x + th, blockDim_x, job, init);
}

void ray_kernel(int threadIdx_x, int blockIdx_x, int blockDim_x, job_t job, int depth, scene_t scene)
{
	int uniq_id = threadIdx_x + blockIdx_x * blockDim_x;

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

void ray_kernel_th(int th_max, int blockIdx_x, int blockDim_x, job_t job, int depth, scene_t scene)
{
	for(int th = 0; th < th_max; th++)
		for(int th_x = 0; th_x < blockDim_x; th_x++)
			ray_kernel(th_x, blockIdx_x + th, blockDim_x, job, depth, scene);
}

void forward_kernel(int threadIdx_x, int blockIdx_x, int blockDim_x, job_t old_job, job_t new_job)
{
	int uniq_id = threadIdx_x + blockIdx_x * blockDim_x;
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

void forward_kernel_th(int th_max, int blockIdx_x, int blockDim_x, job_t old_job, job_t new_job)
{
	for(int th = 0; th < th_max; th++)
		for(int th_x = 0; th_x < blockDim_x; th_x++)
			forward_kernel(th_x, blockIdx_x + th, blockDim_x, old_job, new_job);
}

void rand_kernel(int threadIdx_x, int blockIdx_x, int blockDim_x, job_t old_job, job_t new_job, rand_init_t rand_init)
{
	int uniq_id = threadIdx_x + blockIdx_x * blockDim_x;
	int dest_id = old_job.target_idx[uniq_id] - old_job.gather_arr[uniq_id];
	int max_id = dest_id + old_job.gather_arr[uniq_id];
	float rv[3];

	fast_random_t frand;
	init_fast_random (frand, uniq_id, dest_id, rand_init);

	// remember that the PPS will start with 1
	for(int dest_idx = dest_id; dest_idx < max_id; dest_idx++)
	{
		init_vec3(rv, rand_f(frand), rand_f(frand), rand_f(frand));
		add(&new_job.ray_dir[dest_idx * 3], rv);
	}
}

void rand_kernel_th(int th_max, int blockIdx_x, int blockDim_x, job_t old_job, job_t new_job, rand_init_t rand_init)
{
	for(int th = 0; th < th_max; th++)
		for(int th_x = 0; th_x < blockDim_x; th_x++)
		rand_kernel(th_x, blockIdx_x + th, blockDim_x, old_job, new_job, rand_init);
}

void backward_kernel(int threadIdx_x, int blockIdx_x, int blockDim_x, job_t target, job_t prev)
{
	int uniq_id = threadIdx_x + blockIdx_x * blockDim_x;
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

void backward_kernel_th(int th_max, int blockIdx_x, int blockDim_x, job_t target, job_t prev)
{
	for(int th = 0; th < th_max; th++)
		for(int th_x = 0; th_x < blockDim_x; th_x++)
		backward_kernel(th_x, blockIdx_x + th, blockDim_x, target, prev);
}

void pps_kernel(int threadIdx_x, int blockIdx_x, int blockDim_x, int *dest, int *src, int powerof2)
{
	int uniq_id = threadIdx_x + blockIdx_x * blockDim_x;

	if(uniq_id >= powerof2)
		dest[uniq_id] = src[uniq_id - powerof2] + src[uniq_id];
	else
		dest[uniq_id] = src[uniq_id];
}

void pps_kernel_th(int th_max, int blockIdx_x, int blockDim_x, int *dest, int *src, int powerof2)
{
	for(int th = 0; th < th_max; th++)
		for(int th_x = 0; th_x < blockDim_x; th_x++)
		pps_kernel(th_x, blockIdx_x + th, blockDim_x, dest, src, powerof2);
}

static void do_pps(int *arr, int size)
{
	std::cout << "[PPS] >> For size: " << size * 4 << " B" << std::endl;
	int d_max = ceil_log2(size);
	int *temp = NULL;
	cudaSafeMalloc((void **) &temp, sizeof(int) * size);
	int *orig_temp = temp;

	assert((BLOCKS_PER_JOB(size) % 16) == 0);
	std::cout << "BLOCKS_PER_JOB(size): " << BLOCKS_PER_JOB(size) << std::endl;

	int th_n = MIN(BLOCKS_PER_JOB(size), 16);
	int loop_th = th_n < 16 ? 1 : (BLOCKS_PER_JOB(size) / 16);

	for(int d = 0; d <= d_max; d++)
	{
		std::thread *threads = new std::thread[th_n];

		for(int blk_id = 0; blk_id < th_n; blk_id++)
			threads[blk_id] = std::thread(pps_kernel_th, loop_th, blk_id * loop_th, THREADS_PER_BLOCK, temp, arr, pow2(d));

		for(int blk_id = 0; blk_id < th_n; blk_id++)
			threads[blk_id].join();

		delete [] threads;

		int *swap = temp;
		temp = arr;
		arr = swap;
	}
	if((~d_max) & 1)
	{
		memcpy(arr, temp, size * sizeof(int));
	}

	cudaSafeFree(orig_temp);
}

static int ray_step(job_t dev_job, scene_t *scene, int depth)
{
	int size = calc_jobs(dev_job.image_width * dev_job.image_height);
	assert(size > 0);

	assert((BLOCKS_PER_JOB(size) % 16) == 0);
	std::cout << "BLOCKS_PER_JOB(size): " << BLOCKS_PER_JOB(size) << std::endl;

	int th_n = MIN(BLOCKS_PER_JOB(size), 16);
	int loop_th = th_n < 16 ? 1 : (BLOCKS_PER_JOB(size) / 16);

	if(depth == 0)
	{
		// we have to set the first pos to be that of screen
		std::cout << "[MainLoop] >> Running init_kernel to setup job seed." << std::endl;
		rand_init_t rand_init;
		init_rand(rand_init);

		std::thread *threads = new std::thread[th_n];

		for(int blk_id = 0; blk_id < th_n; blk_id++)
			threads[blk_id] = std::thread(init_kernel_th, loop_th, blk_id * loop_th, THREADS_PER_BLOCK, dev_job, rand_init);

		for(int blk_id = 0; blk_id < th_n; blk_id++)
			threads[blk_id].join();

		delete [] threads;
	}

	std::cout << "[MainLoop] >> Ray tracing kernel .. " << std::endl;
	std::thread *threads = new std::thread[th_n];

	for(int blk_id = 0; blk_id < th_n; blk_id++)
		threads[blk_id] = std::thread(ray_kernel_th, loop_th, blk_id * loop_th, THREADS_PER_BLOCK, dev_job, depth, *scene);

	for(int blk_id = 0; blk_id < th_n; blk_id++)
		threads[blk_id].join();

	delete [] threads;

	int next_size = 0;
	do_pps(dev_job.target_idx, size);
	memcpy(&next_size, &dev_job.target_idx[size - 1], sizeof(int));

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
			int old_jobs_size = calc_jobs(curr_job.image_width * curr_job.image_height);
			std::cout << "[MainLoop newJob] size: " << calc_jobs(temp_job.image_width * temp_job.image_height) << std::endl;

			rand_init_t rand_init;
			init_rand(rand_init);

			assert((BLOCKS_PER_JOB(old_jobs_size) % 16) == 0);
			std::cout << "BLOCKS_PER_JOB(size): " << BLOCKS_PER_JOB(old_jobs_size) << std::endl;

			int th_n = MIN(BLOCKS_PER_JOB(old_jobs_size), 16);
			int loop_th = th_n < 16 ? 1 : (BLOCKS_PER_JOB(old_jobs_size) / 16);

			std::thread *threads = new std::thread[th_n];

			for(int blk_id = 0; blk_id < th_n; blk_id++)
				threads[blk_id] = std::thread(forward_kernel_th, loop_th, blk_id * loop_th, THREADS_PER_BLOCK, curr_job, temp_job);

			for(int blk_id = 0; blk_id < th_n; blk_id++)
				threads[blk_id].join();

			for(int blk_id = 0; blk_id < th_n; blk_id++)
				threads[blk_id] = std::thread(rand_kernel_th, loop_th, blk_id * loop_th, THREADS_PER_BLOCK, curr_job, temp_job, rand_init);

			for(int blk_id = 0; blk_id < th_n; blk_id++)
				threads[blk_id].join();

			delete [] threads;

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

			assert((BLOCKS_PER_JOB(size) % 16) == 0);
			std::cout << "BLOCKS_PER_JOB(size): " << BLOCKS_PER_JOB(size) << std::endl;

			int th_n = MIN(BLOCKS_PER_JOB(size), 16);
			int loop_th = th_n < 16 ? 1 : (BLOCKS_PER_JOB(size) / 16);

			std::thread *threads = new std::thread[th_n];

			for(int blk_id = 0; blk_id < th_n; blk_id++)
				threads[blk_id] = std::thread(backward_kernel_th, loop_th, blk_id * loop_th, THREADS_PER_BLOCK, old_job, back_job);

			for(int blk_id = 0; blk_id < th_n; blk_id++)
				threads[blk_id].join();

			delete [] threads;
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
