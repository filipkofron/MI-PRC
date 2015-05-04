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

int DEPTH_MAX = 1;
#define MIN(a, b) ((a) < (b) ? (a) : (b))
float *const_mem = NULL;

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

void ray_kernel(int threadIdx_x, int blockIdx_x, int blockDim_x, job_t job, scene_t scene, rand_init_t init, uint32_t depth_max)
{
	int uniq_id = threadIdx_x + blockIdx_x * blockDim_x;

	// All threads do work but some of the results will be discarded as they are only the padding in the arrays.
	int kernel_x = uniq_id % job.image_width;
	int kernel_y = uniq_id / job.image_width;

	fast_random_t rand;
	init_fast_random (rand, kernel_x, kernel_y, init);

	trace_ray(
		&job.image_dest[uniq_id * 3],	// assign the result color array
		&job.ray_pos[uniq_id * 3],		// assign job ray position
		&job.ray_dir[uniq_id * 3],		// assign job ray direction
		0,														// this shall stop the recursion
		depth_max,
		&scene,												// const scene
		&rand);
		clamp(&job.image_dest[uniq_id * 3]);
}

void init_kernel_th(int th_max, int blockIdx_x, int blockDim_x, job_t job, rand_init_t init)
{
	for(int th = 0; th < th_max; th++)
		for(int th_x = 0; th_x < blockDim_x; th_x++)
			init_kernel(th_x, blockIdx_x + th, blockDim_x, job, init);
}

void ray_kernel_th(int th_max, int blockIdx_x, int blockDim_x, job_t job, scene_t scene, rand_init_t init, uint32_t depth_max)
{
	for(int th = 0; th < th_max; th++)
		for(int th_x = 0; th_x < blockDim_x; th_x++)
			ray_kernel(th_x, blockIdx_x + th, blockDim_x, job, scene, init, depth_max);
}

void main_loop(job_t host_job, scene_t *scene)
{
	std::cout << "[MainLoop] >> Begin with image size: " << host_job.image_width << "x" << host_job.image_height << " for " << host_job.pass_count << " passes" << std::endl;

	job_t temp_job = host_job;
	job_t dev_job = allocate_device_job(temp_job);

	int size = calc_jobs(dev_job.image_width * dev_job.image_height);
	rand_init_t rand_init;
	init_rand(rand_init);

	assert((BLOCKS_PER_JOB(size) % 16) == 0);
	std::cout << "BLOCKS_PER_JOB(size): " << BLOCKS_PER_JOB(size) << std::endl;

	int th_n = MIN(BLOCKS_PER_JOB(size), 16);
	int loop_th = th_n < 16 ? 1 : (BLOCKS_PER_JOB(size) / 16);

	std::cout << "th_n: " << th_n << std::endl;
	std::cout << "loop_th: " << loop_th << std::endl;

	std::thread *threads = new std::thread[th_n];

	for(int blk_id = 0; blk_id < th_n; blk_id++)
		threads[blk_id] = std::thread(init_kernel_th, loop_th, blk_id * loop_th, THREADS_PER_BLOCK, dev_job, rand_init);

	for(int blk_id = 0; blk_id < th_n; blk_id++)
		threads[blk_id].join();

	for(int blk_id = 0; blk_id < th_n; blk_id++)
		threads[blk_id] = std::thread(ray_kernel_th, loop_th, blk_id * loop_th, THREADS_PER_BLOCK, dev_job, *scene, rand_init, DEPTH_MAX);

	for(int blk_id = 0; blk_id < th_n; blk_id++)
		threads[blk_id].join();

	delete [] threads;

	copy_job_to_host(&host_job, &dev_job);

	free_device_job(&dev_job);
}
