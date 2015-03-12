#ifndef LIGHT_H
#define LIGHT_H

#include "scene.h"
#include "trace.h"

/*
 0 - 2: pos
 3    : radius
 4 - 6: color
 */

#define LIGHT_POS(light) (&light[0])
#define LIGHT_RADIUS(light) (&light[3])
#define LIGHT_COLOR(light) (&light[4])

#define LIGHT_INDEX(idx,light) (&light[LIGHT_SIZE * idx])

#define LIGHT_SIZE 7

void calc_light(float *ray_pos, float *obj_normal, float *light_res, scene_t *scene, color_t *colors);

#endif
