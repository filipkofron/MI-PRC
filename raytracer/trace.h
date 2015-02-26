#ifndef TRACE_H
#define TRACE_H

#include "vec3.h"

void trace_ray(float *color, float *pos, float *dir);
void trace_rect(float *dest, int xs, int ys, int ws, int hs, int w, int h);

#endif
