#ifndef COMMON_CUH
#define COMMON_CUH

#include <cstdio>
#include <iostream>
#include <chrono>

void cudaSafeMalloc(void **ptr, size_t size);
void safeMalloc(void **ptr, size_t size);
void cudaSafeFree(void *ptr);
void safeFree(void *ptr);
int ceil_log2(unsigned long long x);
int pow2(int e);

class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const {
        return std::chrono::duration_cast<second_>
            (clock_::now() - beg_).count(); }

private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};

#endif
