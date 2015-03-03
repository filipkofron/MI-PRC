#ifndef TRACE_H
#define TRACE_H

#include "vec3.h"
#include "rand.hpp"

struct color_t
{
        float ambient[3];
        float diffuse[3];
        float specular[3];
};

struct scene_t
{
        float *spheres;
        int spheres_count;
        float *triangles;
        int triangles_count;
};

extern scene_t scene;

void trace_ray(
        float *color,
        float *pos,
        float *dir,
        uint32_t depth,
        FastRandom &random);
void trace_rect(float *dest, int xs, int ys, int ws, int hs, int w, int h);

#endif
