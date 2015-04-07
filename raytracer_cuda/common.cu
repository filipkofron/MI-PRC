#include "common.cuh"

void cudaSafeMalloc(void **ptr, size_t size)
{
  if (cudaMalloc(&ptr, size) != cudaSuccess)
	{
		std::cerr << "Cannot allocate memory of size " << (size / 1024) << " kiB on device!" << std::endl;
		exit(1);
	}
}

void safeMalloc(void **ptr, size_t size)
{
  *ptr = malloc(size);
  if (*ptr == NULL)
  {
    std::cerr << "Cannot allocate memory of size " << (size / 1024) << " kiB on host!" << std::endl;
    exit(1);
  }
}

void cudaSafeFree(void *ptr)
{
  cudaFree(ptr);
}

void safeFree(void *ptr)
{
  free(ptr);
}
