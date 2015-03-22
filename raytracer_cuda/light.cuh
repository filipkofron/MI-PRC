#ifndef LIGHT_H
#define LIGHT_H

#include "scene.cuh"
#include "trace.cuh"

/*
byte positions
0 - 2: pos
3    : radius
4 - 6: color
*/

// Get ptr to light pos
#define LIGHT_POS(light) (&light[0])

// get light raidus ptr
#define LIGHT_RADIUS(light) (&light[3])

// get light color ptr
#define LIGHT_COLOR(light) (&light[4])

// get ptr to index in lights array
#define LIGHT_INDEX(idx,light) (&light[LIGHT_SIZE * idx])

#define LIGHT_SIZE 7

/*
calculate light at a given position of ray intersect
*/
void calc_light(float *ray_pos, float *obj_normal, float *light_res, scene_t *scene, color_t *colors);

#endif
