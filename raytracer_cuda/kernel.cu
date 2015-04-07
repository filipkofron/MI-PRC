
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "bmp.cuh"
#include "scene.cuh"
#include "trace.cuh"

#include <cstdio>
#include <iostream>

#define TEST_WIDTH 4096
#define TEST_HEIGHT 4096

static void HandleError(cudaError_t err, const char *file, int line)
{
	if (err != cudaSuccess)
	{
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__global__ void ray_kernel(float *result_image, int divB, int sizeB, int ws, int hs, int width, int height, scene_t device_scene)
{
	int x_t = threadIdx.x;
	int y_b = blockIdx.x;
	//trace_rect(result_image, x, y, ws, hs, width, height, &device_scene);

	for(int x = x_t * ws; x < (x_t + 1) * ws; x++)
		for(int y = y_b * hs; y < (y_b + 1) * hs; y++)
			trace_rect(result_image, x, y, ws, hs, width, height, &device_scene);
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

	int blocks = 512;
	int threads = 512;

	int ws = TEST_WIDTH / blocks;
	int hs = TEST_HEIGHT / threads;

	ray_kernel <<< blocks, threads >>>(cuda_result_image, blocks, threads, ws, hs, TEST_WIDTH, TEST_HEIGHT, dev_scene);
	cudaDeviceSynchronize();

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

//	std::cout << "All done, press any key to exit..." << std::endl;
//	std::cin.get();

    return 0;
}
