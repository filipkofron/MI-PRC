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

int DEPTH_MAX = 1;

float __constant__ const_mem[15 * 1024];

__global__ void init_kernel(job_t job, rand_init_t init)
{
	int uniq_id = threadIdx.x + blockIdx.x * blockDim.x;

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

__global__ void ray_kernel(job_t job, scene_t scene, rand_init_t init, uint32_t depth_max)
{
	int uniq_id = threadIdx.x + blockIdx.x * blockDim.x;

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

void main_loop(job_t host_job, scene_t *scene)
{
	std::cout << "[MainLoop] >> Begin with image size: " << host_job.image_width << "x" << host_job.image_height << " for " << host_job.pass_count << " passes" << std::endl;

	job_t temp_job = host_job;
	job_t dev_job = allocate_device_job(temp_job);

	int size = calc_jobs(dev_job.image_width * dev_job.image_height);
	rand_init_t rand_init;
	init_rand(rand_init);

	init_kernel<<< BLOCKS_PER_JOB(size), THREADS_PER_BLOCK >>>(dev_job, rand_init);
	ray_kernel<<< BLOCKS_PER_JOB(size), THREADS_PER_BLOCK >>>(dev_job, *scene, rand_init, DEPTH_MAX);
	cudaDeviceSynchronize();
	cudaCheckErrors("ray_kernel fail");

	copy_job_to_host(&host_job, &dev_job);

	free_device_job(&dev_job);
}
