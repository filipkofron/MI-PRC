#ifndef CIRCLE_H
#define CIRCLE_H

#include <cfloat>
#include "vec3.h"

/*
0 - 2: pos
3:        radius
4 - 6:  Ambient color
7 - 9: Diffuser color
10 - 12: Specular color
13:     transparency
*/

#define SPHERE_POS(spheres) (&spheres[0])
#define SPHERE_RADIUS(spheres) (&spheres[3])
#define SPHERE_AMBIENT(spheres) (&spheres[4])
#define SPHERE_DIFFUSE(spheres) (&spheres[7])
#define SPHERE_SPECULAR(spheres) (&spheres[10])
#define SPHERE_TRANSPARENCY(spheres) (&spheres[13])

#define SPHERE_SIZE (14)
#define SPHERE_INDEX(idx,spheres) (&spheres[SPHERE_SIZE * idx])

/* calculate possible intersect */
inline float sphere_intersect(float *pos, float *dir, float *center, float r)
{
	float ray_pos_x = pos[0] - center[0];
	float ray_pos_y = pos[1] - center[1];
	float ray_pos_z = pos[2] - center[2];

	float b = ray_pos_x * dir[0] + ray_pos_y * dir[1] + ray_pos_z * dir[2];
	float t = b * b - ray_pos_x * ray_pos_x - ray_pos_y * ray_pos_y - ray_pos_z * ray_pos_z + r * r;

	float sol;

	if (t < 0)
		return FLT_MAX;

	sol = -b - sqrt(t);

	if (sol > 0)
		return sol * 0.9999f;

	sol = -b + sqrt(t);

	if (sol > 0)
		return sol * 0.9999f;

	return FLT_MAX;
}

/* retrieve position of the intersect */
inline void sphere_intersect_pos(float *new_pos, float *old_pos, float *old_dir, float dist)
{
	new_pos[0] = old_pos[0] + old_dir[0] * dist;
	new_pos[1] = old_pos[1] + old_dir[1] * dist;
	new_pos[2] = old_pos[2] + old_dir[2] * dist;
}

/* calculare sphere normal at given position on the sphere */
inline void sphere_normal(float *normal, float *pos, float *sphere)
{
	normal[0] = pos[0] - sphere[0];
	normal[1] = pos[1] - sphere[1];
	normal[2] = pos[2] - sphere[2];

	normalize(normal);
}

#endif
