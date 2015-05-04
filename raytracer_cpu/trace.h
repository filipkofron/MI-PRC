#ifndef TRACE_H
#define TRACE_H

#include "color.cuh"
#include "vec3.cuh"
#include "scene.cuh"

__device__ void trace_ray(
	int *gather_arr,
	int *target_idx,
	float *color,
	float *pos,
	float *dir,
	uint32_t depth,
	scene_t *scene);

__device__ int find_intersect(float *pos, float *dir, float *new_pos, float *new_dir, float *normal, color_t *colors, scene_t *scene);
//__device__ void trace_rect(float *dest, int xs, int ys, int ws, int hs, int w, int h, scene_t *scene);

#endif
