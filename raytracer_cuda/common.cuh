#ifndef COMMON_CUH
#define COMMON_CUH

void cudaSafeMalloc(void **ptr, size_t size);
void safeMalloc(void **ptr, size_t size);
void cudaSafeFree(void *ptr);
void safeFree(void *ptr);

#endif
