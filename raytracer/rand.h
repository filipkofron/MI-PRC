#ifndef RAND_HPP
#define RAND_HPP

#include <stdint.h>
#include <mutex>

/* These state variables must be initialized so that they are not all zero. */

static std::mutex mutex;

class FastRandom
{
protected:
    uint32_t x, y, z, w;

public:
    FastRandom()
    {
        std::unique_lock<std::mutex>(mutex);

        while(x == 0 || y == 0 || z == 0 || w == 0)
        {
            x = (uint32_t) rand();
            y = (uint32_t) rand();
            z = (uint32_t) rand();
            w = (uint32_t) rand();
        }
    }

    inline uint32_t rand_full(void)
    {
        uint32_t t = x ^(x << 11);
        x = y;
        y = z;
        z = w;
        return w = w ^ (w >> 19) ^ t ^ (t >> 8);
    }

    inline uint32_t rand256(void)
    {
        uint32_t t = x ^(x << 11);
        x = y;
        y = z;
        z = w;
        return (w = w ^ (w >> 19) ^ t ^ (t >> 8)) & 255;
    }
};

#endif
