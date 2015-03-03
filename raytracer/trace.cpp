#include <stdint.h>
#include "trace.h"
#include "sphere.h"
#include "triangle.hpp"

#define DEPTH_MAX 3

#define RANDOM_MOVE_VEC(vec,random) \
do {\
vec [0] += 0.5f / (random.rand256() - 127); \
vec [1] += 0.5f / (random.rand256() - 127); \
vec [2] += 0.5f / (random.rand256() - 127); \
} while (false)

scene_t scene;

int find_intersect(float *pos, float *dir, float *new_pos, float *new_dir, float *normal, color_t *colors)
{
    float dist_tr = FLT_MAX;
    float dist_sp = FLT_MAX;
    int closest_tr = -1;
    int closest_sp = -1;
    int sp_tr_none = -1;
    int res = 0;

    float tr_uv[2];
    for(uint32_t i = 0; i < scene.spheres_count; i++)
    {
        float temp = sphere_intersect(pos, dir, SPHERE_POS(SPHERE_INDEX(i, scene.spheres)), *SPHERE_RADIUS(SPHERE_INDEX(i, scene.spheres)));
        if(temp < dist_sp)
        {
            dist_sp = temp;
            closest_sp = i;
        }
    }
    for(uint32_t i = 0; i < scene.triangles_count; i++)
    {
        float temp = triangle_intersect(pos, dir, TRIANGLE_POS(TRIANGLE_INDEX(i, scene.triangles)), tr_uv);
        if(temp < dist_tr)
        {
            dist_tr = temp;
            closest_tr = i;
        }
    }


    if(closest_sp != -1 && closest_tr != -1)
    {
        sp_tr_none = dist_sp <= dist_tr ? 0 : 1;
    }
    else
    {
        if(closest_sp != -1)
        {
            sp_tr_none = 0;
        }
        if(closest_tr != -1)
        {
            sp_tr_none = 1;
        }
    }

    float *elem;
    switch(sp_tr_none)
    {
        case 0:
            sphere_intersect_pos(new_pos, pos, dir, dist_sp);
            elem = SPHERE_INDEX(closest_sp, scene.spheres);
            sphere_normal(normal, new_pos, SPHERE_POS(elem));
            set_vec3(colors->ambient, SPHERE_AMBIENT(elem));
            set_vec3(colors->diffuse, SPHERE_DIFFUSE(elem));
            set_vec3(colors->specular, SPHERE_SPECULAR(elem));
            res = 1;
            break;
        case 1:
            elem = TRIANGLE_INDEX(closest_tr, scene.triangles);
            triangle_pos(new_pos, tr_uv, TRIANGLE_POS(elem));
            triangle_normal(normal, TRIANGLE_POS(elem));
            set_vec3(colors->ambient, TRIANGLE_AMBIENT(elem));
            set_vec3(colors->diffuse, TRIANGLE_DIFFUSE(elem));
            set_vec3(colors->specular, TRIANGLE_SPECULAR(elem));
            set_vec3(colors->specular, TRIANGLE_SPECULAR(elem));
            res = 1;
            break;
    }

    if(res)
    {
        reflection(new_dir, dir, normal);
    }

    return res;
}

void trace_ray(
        float *color,
        float *pos,
        float *dir,
        uint32_t depth,
        FastRandom &random)
{
    float new_pos[3];
    float new_dir[3];
    float normal[3];
    float none[3] = {0, 0, 0};
    color_t colors;

    if(find_intersect(pos, dir, new_pos, new_dir, normal, &colors))
    {
        set_vec3(color, pos);
    }
    else
    {
        set_vec3(color, none);
    }
}

void trace_rect(float *dest, int xs, int ys, int ws, int hs, int w, int h)
{
    float pos[3];
    float dir[3];

    FastRandom random;

    for(int x = xs; x < (xs + ws); x++)
    {
        for(int y = ys; y < (ys + hs); y++)
        {
            init_vec3(pos, x - 0.5f * w, y - 0.5f * h, 0.0f);
            float off_x = pos[0] * 0.001f;
            float off_y = pos[1] * 0.001f;
            init_vec3(dir, off_x, off_y, 1.0f);
            normalize(dir);
            float *color_offset = &dest[(y * w + x) * 3];
            color_offset[0] = color_offset[1] = color_offset[2] = 0.0f;
            float temp_color[3];
            const int num = 3;
            for(int i = 0; i < num; i++)
            {
                trace_ray(temp_color, pos, dir, 0, random);
                add(color_offset, temp_color);
            }
            mul(color_offset, 1.0f / num);

            if(color_offset[0] > 1.0f)
            {
                color_offset[0] = 1.0f;
            }
            if(color_offset[1] > 1.0f)
            {
                color_offset[1] = 1.0f;
            }
            if(color_offset[2] > 1.0f)
            {
                color_offset[2] = 1.0f;
            }
        }
    }
}
