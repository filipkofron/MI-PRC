#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include "kernel.cuh"
#include "bmp.cuh"
#include "common.cuh"
#include "jpge.h"

#include <cstdio>
#include <iostream>
#include <sstream>

#include <unistd.h>

void print_usage()
{
	std::cout << "Usage:" << std::endl;
	std::cout << "raytracer IMAGE_WIDTH IMAGE_HEIGHT PASS_COUNT DEPTH" << std::endl;
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

void init_cuda()
{
	cuInit(0);
	cudaCheckErrors("cuInit fail");
	int num_devices, device;
	cudaGetDeviceCount(&num_devices);
	if (num_devices > 1)
	{
		int max_multiprocessors = 0, max_device = 0;
		for (device = 0; device < num_devices; device++) {
			cudaDeviceProp properties;
			cudaGetDeviceProperties(&properties, device);
			cudaCheckErrors("cudaGetDeviceProperties fail");
			if (max_multiprocessors < properties.multiProcessorCount)
			{
	      max_multiprocessors = properties.multiProcessorCount;
	      max_device = device;
			}
		}
		cudaSetDevice(max_device);
		cudaCheckErrors("cudaSetDevice fail");

		CUdevice dev;
		cuDeviceGet(&dev, max_device);
		cudaCheckErrors("cuDeviceGet fail.");

		CUcontext ctx;
		cuCtxCreate(&ctx, 0, dev);
		cudaCheckErrors("cuCtxCreate fail.");
	}
	else
	{
		std::cerr << "No CUDA device abailable :(." << std::endl;
		exit(1);
	}
}

int main(int argc, char *argv[])
{
	srand(time(NULL));

	init_cuda();
	cudaDeviceSynchronize();
	cudaCheckErrors("Init fail.");

	const clock_t whole_time_begin = clock();
	if(argc != 5)
	{
		std::cerr << "Invalid number of arguments: " << (argc - 1) << " but 4 are required." << std::endl;
    print_usage();
		exit(1);
	}

	job_t host_job;

	std::stringstream ss;
	ss << argv[1] << " " << argv[2] << " " << argv[3] << " " << argv[4];

	if(!(ss >> host_job.image_width)) { print_usage(); exit(1); }
	if(!(ss >> host_job.image_height)) { print_usage(); exit(1); }
	if(!(ss >> host_job.pass_count)) { print_usage(); exit(1); }
	if(!(ss >> DEPTH_MAX)) { print_usage(); exit(1); }

	host_job = allocate_host_job(host_job);

	cudaCheckErrors("Some fail.");

	cudaDeviceSynchronize();

	cudaCheckErrors("Some fail #2.");

	init_scene("sample/sample", host_job.image_width, host_job.image_height);

	cudaCheckErrors("Scene fail.");

	std::cout << "[Prep] >> Done." << std::endl;

	const clock_t cuda_time_begin = clock();

	main_loop(host_job, &dev_scene);
	cudaDeviceSynchronize();

	float res_cuda = (float( clock () - cuda_time_begin ) / CLOCKS_PER_SEC) * 1000;

	clean_scene();

	/*FILE *file = fopen("test.bmp", "wb+");
	if (file)
	{
		write_bmp(file, host_job.image_dest, host_job.image_width, host_job.image_height);
		fflush(file);
		fclose(file);
	}
	else
	{
		fprintf(stderr, "File could not be opened!\n");
	}*/

	uint8_t *img = new uint8_t[host_job.image_width * host_job.image_height * 3];
	int max_size = host_job.image_width * host_job.image_height * 3;
	for (uint32_t y = 0; y < host_job.image_height; y++)
	{
		for (uint32_t x = 0; x < host_job.image_width; x++)
		{
			float *elem = &host_job.image_dest[max_size - (y * host_job.image_width + x) * 3 - 1];
			uint8_t *dest = &img[(y * host_job.image_width + (host_job.image_width - x - 1)) * 3];

			dest[2] = (uint8_t)(elem[0] * 255.0);
			dest[1] = (uint8_t)(elem[1] * 255.0);
			dest[0] = (uint8_t)(elem[2] * 255.0);
		}
	}

	if(!jpge::compress_image_to_jpeg_file("image.jpg", host_job.image_width, host_job.image_height, 3, img))
	{
		fprintf(stderr, "Could not save JPEG!\n");
	}

	delete [] img;

	free_host_job(&host_job);

	cudaError_t cudaStatus = cudaDeviceReset();
  if (cudaStatus != cudaSuccess)
	{
      fprintf(stderr, "cudaDeviceReset failed!");
      return 1;
  }

	float res_whole = (float( clock () - whole_time_begin ) / CLOCKS_PER_SEC) * 1000;

	std::cout << std::endl;
	std::cout << "[Post] >> Overall time in ms: " << res_whole << std::endl;
	std::cout << "[Post] >> CUDA time in ms: " << res_cuda << std::endl;
	std::cout << "[Post] >> All done." << std::endl;

	return 0;
}
