#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.cuh"
#include "bmp.cuh"

#include <cstdio>
#include <iostream>

void print_usage()
{
	std::cout << "Usage:" << std::endl;
	std::cout << "raytracer IMAGE_WIDTH IMAGE_HEIGHT PASS_COUNT" << std::endl;
	std::cout << "PASS_COUNT is the number of passes to average random differences." << std::endl;
	std::cout << std::endl;
}


static void HandleError(cudaError_t err, const char *file, int line)
{
	if (err != cudaSuccess)
	{
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

int main(int argc, char *argv[])
{
	if(argc != 4)
	{
		std::cerr << "Invalid number of arguments: " << (argc - 1) << " but 3 are required." << std::endl;
    print_usage();
		exit(1);
	}

	job_t host_job;

	if(!(std::cin >> host_job.image_width)) print_usage();
	if(!(std::cin >> host_job.image_height)) print_usage();
	if(!(std::cin >> host_job.pass_count)) print_usage();

	int size = host_job.image_width * host_job.image_height * 3;
	float *host_result_image = (float *)malloc(sizeof(float) * size);
	float *cuda_result_image;

	if (cudaMalloc(&cuda_result_image, sizeof(float)* size) != cudaSuccess)
	{
		std::cerr << "Cannot allocate memory for result image on device!" << std::endl;
		exit(1);
	}

	init_scene("sample/sample", TEST_WIDTH, TEST_HEIGHT);

	std::cout << "[Prep] >> Done." << std::endl;

	int blocks = 128;
	int threads = 128;

	//ray_kernel <<< blocks, threads >>>();
	cudaDeviceSynchronize();

	clean_scene();

	if (cudaMemcpy(host_result_image, cuda_result_image, sizeof(float) * size, cudaMemcpyDeviceToHost) != cudaSuccess)
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

	cudaError_t cudaStatus = cudaDeviceReset();
  if (cudaStatus != cudaSuccess)
	{
      fprintf(stderr, "cudaDeviceReset failed!");
      return 1;
  }

		std::cout << "All done." << std::endl;

		return 0;
}
