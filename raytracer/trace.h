#ifndef TRACE_H
#define TRACE_H

#include "vec3.h"
#include "rand.hpp"

void trace_ray(float *color, float *pos, float *dir, float *spheres, uint32_t spheres_count, uint32_t depth, FastRandom &random);
void trace_rect(float *dest, int xs, int ys, int ws, int hs, int w, int h);

#endif
