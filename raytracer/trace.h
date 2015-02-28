#ifndef TRACE_H
#define TRACE_H

#include "vec3.h"

void trace_ray(float *color, float *pos, float *dir, float *spheres, uint32_t spheres_count, uint32_t depth);
void trace_rect(float *dest, int xs, int ys, int ws, int hs, int w, int h);

#endif
