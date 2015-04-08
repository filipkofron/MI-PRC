#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.cuh"
#include "bmp.cuh"

#include <cstdio>
#include <iostream>
#include <sstream>

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

	std::stringstream ss;
	ss << argv[1] << " " << argv[2] << " " << argv[3];

	if(!(ss >> host_job.image_width)) { print_usage(); exit(1); }
	if(!(ss >> host_job.image_height)) { print_usage(); exit(1); }
	if(!(ss >> host_job.pass_count)) { print_usage(); exit(1); }

	host_job = allocate_host_job(host_job);

	init_scene("sample/sample", host_job.image_width, host_job.image_height);

	std::cout << "[Prep] >> Done." << std::endl;

	//main_loop(host_job, &dev_scene);
	cudaDeviceSynchronize();

	clean_scene();

	FILE *file = fopen("test.bmp", "wb+");
	srand((unsigned int) time(NULL));
	if (file)
	{
		write_bmp(file, host_job.image_dest, host_job.image_width, host_job.image_height);
		fflush(file);
		fclose(file);
	}
	else
	{
		fprintf(stderr, "File could not be opened!\n");
	}

	free_host_job(&host_job);

	cudaError_t cudaStatus = cudaDeviceReset();
  if (cudaStatus != cudaSuccess)
	{
      fprintf(stderr, "cudaDeviceReset failed!");
      return 1;
  }

		std::cout << "All done." << std::endl;

		return 0;
}
