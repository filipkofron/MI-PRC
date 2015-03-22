#ifndef TRIANGLE_HPP
#define TRIANGLE_HPP

#include <float.h>
#include "vec3.cuh"

/*
0 - 2: pos a
3 - 5: pos b
6 - 8: pos c
9 - 11:  Ambient color
12 - 14: Diffuser color
15 - 17: Specular color
18:     transparency
*/

#define TRIANGLE_POS(triangles) (&triangles[0])
#define TRIANGLE_POS_A(triangles) (&triangles[0])
#define TRIANGLE_POS_B(triangles) (&triangles[3])
#define TRIANGLE_POS_C(triangles) (&triangles[6])
#define TRIANGLE_AMBIENT(triangles) (&triangles[9])
#define TRIANGLE_DIFFUSE(triangles) (&triangles[12])
#define TRIANGLE_SPECULAR(triangles) (&triangles[15])
#define TRIANGLE_TRANSPARENCY(triangles) (&triangles[18])

#define TRIANGLE_SIZE (19)
#define TRIANGLE_INDEX(idx,triangles) (&triangles[TRIANGLE_SIZE * idx])


inline float triangle_intersect(float *pos, float *dir, float *triangle, float *res_uv)
{
	float e1[3], e2[3], h[3], s[3], q[3];
	float a, f, u, v;

	sub(e1, &triangle[3], &triangle[0]);
	sub(e2, &triangle[6], &triangle[0]);

	cross(h, dir, e2);

	a = dot(e1, h);

	if (a > -0.00001f && a < 0.00001f)
		return FLT_MAX;

	f = 1.0f / a;

	sub(s, pos, &triangle[0]);

	u = f * dot(s, h);

	if (u < 0.0f || u > 1.0f)
		return FLT_MAX;

	cross(q, s, e1);
	v = f * dot(dir, q);

	if (v < 0.0f || u + v > 1.0f)
		return FLT_MAX;

	res_uv[0] = u;
	res_uv[1] = v;

	float dist = f * dot(e2, q);

	return dist > 0.00001f ? dist : FLT_MAX;
}

inline void triangle_pos(float *new_pos, float *uv, float *triangle)
{
	float d = 1 - uv[0] - uv[1];
	float temp[3];

	mul(new_pos, &triangle[0], d);
	mul(temp, &triangle[3], uv[0]);
	add(new_pos, temp);
	mul(temp, &triangle[6], uv[1]);
	add(new_pos, temp);
}

inline void triangle_normal(float *normal, float *triangle)
{
	float e1[3];
	float e2[3];

	sub(e1, &triangle[3], &triangle[0]);
	sub(e2, &triangle[6], &triangle[0]);

	cross(normal, e1, e2);
	normalize(normal);
}

#endif