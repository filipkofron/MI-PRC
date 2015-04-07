#include "cuda_runtime.h"
#include "device_launch_parameters.h"
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

// TODO: PPS over the array
/*
	__shared__ int pps_arr[THREADS_PER_BLOCK];

	int uniq_id = threadIdx.x + blockIdx.x * blockDim.x;

  pps_arr[threadIdx.x] = job.target_idx[uniq_id];
	atomicAdd(&pps_arr[], );

	__syncthreads();*/
}
