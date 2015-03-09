#ifndef TRACE_H
#define TRACE_H

#include "vec3.h"
#include "rand.h"
#include "scene.h"

struct color_t
{
        float ambient[3];
        float diffuse[3];
        float specular[3];
};


extern scene_t scene;

void trace_ray(
        float *color,
        float *pos,
        float *dir,
        uint32_t depth,
        FastRandom &random);

int find_intersect(float *pos, float *dir, float *new_pos, float *new_dir, float *normal, color_t *colors);
void trace_rect(float *dest, int xs, int ys, int ws, int hs, int w, int h);

#endif
