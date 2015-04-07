#include <stdint.h>
#include <iostream>
#include "trace.cuh"
#include "sphere.cuh"
#include "triangle.cuh"
#include "light.cuh"

#define DEPTH_MAX 3

	// checks whole scene for intersect
__device__ int find_intersect(float *pos, float *dir, float *new_pos, float *new_dir, float *normal, color_t *colors, scene_t *scene)
{
		float dist_tr = FLT_MAX;
		float dist_sp = FLT_MAX;
		int closest_tr = -1;
		int closest_sp = -1;
		int sp_tr_none = -1;
		int res = 0;

		float tr_uv[2];
		float temp_tr_uv[2];
		for (int i = 0; i < scene->spheres_count; i++)
		{
			float temp = sphere_intersect(pos, dir, SPHERE_POS(SPHERE_INDEX(i, scene->spheres)), *SPHERE_RADIUS(SPHERE_INDEX(i, scene->spheres)));
			if (temp < dist_sp)
			{
				dist_sp = temp * 0.99999f;
				closest_sp = i;
			}
		}
		for (int i = 0; i < scene->triangles_count; i++)
		{
			float temp = triangle_intersect(pos, dir, TRIANGLE_POS(TRIANGLE_INDEX(i, scene->triangles)), temp_tr_uv);
			if (temp < dist_tr)
			{
				dist_tr = temp * 0.99999f;
				closest_tr = i;
				set_vec3(tr_uv, temp_tr_uv);
			}
		}


		if (closest_sp != -1 && closest_tr != -1)
		{
			sp_tr_none = dist_sp <= dist_tr ? 0 : 1;
		}
		else
		{
			if (closest_sp != -1)
			{
				sp_tr_none = 0;
			}
			if (closest_tr != -1)
			{
				sp_tr_none = 1;
			}
		}

		float *elem;
		switch (sp_tr_none)
		{
		case 0:
			sphere_intersect_pos(new_pos, pos, dir, dist_sp);
			elem = SPHERE_INDEX(closest_sp, scene->spheres);
			sphere_normal(normal, new_pos, SPHERE_POS(elem));
			set_vec3(colors->ambient, SPHERE_AMBIENT(elem));
			set_vec3(colors->diffuse, SPHERE_DIFFUSE(elem));
			set_vec3(colors->specular, SPHERE_SPECULAR(elem));

			res = 1;
			break;
		case 1:
			elem = TRIANGLE_INDEX(closest_tr, scene->triangles);
			mul(new_pos, dir, dist_tr);
			add(new_pos, pos);
			//triangle_pos(new_pos, tr_uv, TRIANGLE_POS(elem));
			triangle_normal(normal, TRIANGLE_POS(elem));
			set_vec3(colors->ambient, TRIANGLE_AMBIENT(elem));
			set_vec3(colors->diffuse, TRIANGLE_DIFFUSE(elem));
			set_vec3(colors->specular, TRIANGLE_SPECULAR(elem));

			res = 1;
			break;
		}

		if (res)
		{
			reflection(new_dir, dir, normal);
		}

		return res;
	}

// trace single ray
__device__ void trace_ray(
	float *color,
	float *pos,
	float *dir,
	uint32_t depth,
	scene_t *scene)
{
	float new_pos[3] = {0.0f, 0.0f, 0.0f};
	float new_dir[3] = {0.0f, 0.0f, 0.0f};
	float normal[3] = {0.0f, 0.0f, 0.0f};
	float none[3] = { 0.12f, 0.1f, 0.11f };
	color_t colors;

set_vec3(color, none);

	if (find_intersect(pos, dir, new_pos, new_dir, normal, &colors, scene))
	{
		normalize(normal);
		float light_color[3] = {0.0f, 0.0f, 1.0f};
		calc_light(new_pos, normal, light_color, scene, &colors);
		set_vec3(color, light_color);
	}
	else
	{
		set_vec3(color, none);
	}
}

// trace rectangle segment
/*__device__ void trace_rect(float *dest, int x, int y, int ws, int hs, int w, int h, scene_t *scene)
{
	float pos[3];
	float dir[3];

	float norm = (w > h ? w : h);

	init_vec3(pos, (x - 0.5f * w) / norm, (y - 0.5f * h) / norm, -60.0f);
	float off_x = pos[0] * 0.001f;
	float off_y = pos[1] * 0.001f;
	init_vec3(dir, off_x, off_y, 1.0f);
	normalize(dir);
	float *color_offset = &dest[(y * w + x) * 3];
	color_offset[0] = color_offset[1] = color_offset[2] = 0.0f;
	float temp_color[3];
	init_vec3(temp_color);
	const int num = 1;
	for (int i = 0; i < num; i++)
	{
		trace_ray(temp_color, pos, dir, 0, scene);
		add(color_offset, temp_color);
	}
	mul(color_offset, 1.0f / num);


}*/
