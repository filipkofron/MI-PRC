#ifndef COMMON_CUH
#define COMMON_CUH

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

void cudaSafeMalloc(void **ptr, size_t size);
void safeMalloc(void **ptr, size_t size);
void cudaSafeFree(void *ptr);
void safeFree(void *ptr);
int ceil_log2(unsigned long long x);
int pow2(int e);

#endif
