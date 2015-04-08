#include "common.cuh"

#include <iostream>

void cudaSafeMalloc(void **ptr, size_t size)
{
  std::cout << "Allocating " << size << " B on device!" << std::endl;
  if (cudaMalloc(ptr, size) != cudaSuccess)
	{
		std::cerr << "Cannot allocate memory of size " << (size / 1024) << " kiB on device!" << std::endl;
		exit(1);
	}
}

void safeMalloc(void **ptr, size_t size)
{
  *ptr = malloc(size);
  std::cout << "Allocating " << size << " B on host!" << std::endl;
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

int ceil_log2(unsigned long long x)
{
  static const unsigned long long t[6] = {
    0xFFFFFFFF00000000ull,
    0x00000000FFFF0000ull,
    0x000000000000FF00ull,
    0x00000000000000F0ull,
    0x000000000000000Cull,
    0x0000000000000002ull
  };

  int y = (((x & (x - 1)) == 0) ? 0 : 1);
  int j = 32;
  int i;

  for (i = 0; i < 6; i++) {
    int k = (((x & t[i]) == 0) ? 0 : j);
    y += k;
    x >>= k;
    j >>= 1;
  }

  return y;
}

int pow2(int e)
{
	if(e < 1)
	{
		return 1;
	}
	return 1 << e;
}
