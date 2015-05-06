#include "kernel.h"
#include "bmp.h"
#include "common.h"
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

int main(int argc, char *argv[])
{
	srand(time(NULL));

	Timer timerWhole;
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

	init_scene("sample/sample", host_job.image_width, host_job.image_height);

	std::cout << "[Prep] >> Done." << std::endl;

	Timer timerCuda;

	main_loop(host_job, &dev_scene);

	double res_cuda = timerCuda.elapsed();

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

	double res_whole = timerWhole.elapsed();

	std::cout << std::endl;
	std::cout << "[Post] >> Overall time in s: " << res_whole << std::endl;
	std::cout << "[Post] >> CUDA time in s: " << res_cuda << std::endl;
	std::cout << "[Post] >> All done." << std::endl;

	return 0;
}
