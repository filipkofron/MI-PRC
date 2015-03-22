
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "bmp.cuh"
#include "scene.cuh"
#include "trace.cuh"

#include <cstdio>
#include <iostream>

#define TEST_WIDTH 1280
#define TEST_HEIGHT 1024

__global__ void ray_kernel(float *result_image, int divB, int sizeB, int ws, int hs, int width, int height, scene_t device_scene)
{
	int global_index = threadIdx.x + blockDim.x * threadIdx.y;

	int x = global_index / 1024;
	int y = global_index % 1024;
	//trace_rect(result_image, x, y, ws, hs, width, height, &device_scene);
	float *color_offset = &result_image[(y * w + x) * 3];
	color_offset[0] = global_index / 1000.0;
	color_offset[1] = global_index / 1000.0;
	color_offset[2] = global_index / 1000.0;
}

int main()
{   
	int size = TEST_WIDTH * TEST_HEIGHT * 3;
	float *host_result_image = (float *)malloc(sizeof(float) * size);
	float *cuda_result_image;
	if (cudaMalloc(&cuda_result_image, sizeof(float)* size) != cudaSuccess)
	{
		std::cerr << "Cannot allocate memory for result image on device!" << std::endl;
		exit(1);
	}

	init_scene("sample/sample", TEST_WIDTH, TEST_HEIGHT);

	std::cout << "[Prep] >> Done." << std::endl;

	int ws = TEST_WIDTH / 1280;
	int hs = TEST_HEIGHT / 1024;

	ray_kernel << < 1280, 1024 >> >(cuda_result_image, 1280, 1024, ws, hs, TEST_WIDTH, TEST_HEIGHT, dev_scene);
	cudaDeviceSynchronize();

	//trace_all(TEST_WIDTH, TEST_HEIGHT, test);
	clean_scene();

	if (cudaMemcpy(host_result_image, cuda_result_image, sizeof(float)* size, cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		std::cerr << "Cannot copy result image from device!" << std::endl;
		exit(1);
	}

	FILE *file = fopen("test.bmp", "wb+");
	srand((unsigned int) time(NULL));
	if (file)
	{
		write_bmp(file, host_result_image, TEST_WIDTH, TEST_HEIGHT);
		fflush(file);
		fclose(file);
	}
	else
	{
		fprintf(stderr, "File could not be opened!\n");
	}
	free(host_result_image);
	cudaFree(cuda_result_image);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

	std::cout << "All done, press any key to exit..." << std::endl;
	std::cin.get();

    return 0;
}
