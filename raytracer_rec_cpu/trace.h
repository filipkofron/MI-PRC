#ifndef TRACE_H
#define TRACE_H

#include "color.h"
#include "vec3.h"
#include "scene.h"
#include "rand.h"

void trace_ray(
	float *color,
	float *pos,
	float *dir,
	uint32_t depth,
	uint32_t depth_max,
	scene_t *scene,
	fast_random_t *frand);

int find_intersect(float *pos, float *dir, float *new_pos, float *new_dir, float *normal, color_t *colors, scene_t *scene);
//__device__ void trace_rect(float *dest, int xs, int ys, int ws, int hs, int w, int h, scene_t *scene);

#endif
