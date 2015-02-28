#include <stdint.h>
#include <stdlib.h>
#include "trace.h"
#include "sphere.h"

#define DEPTH_MAX 3

#define spheres_test_cnt 7

float test_sun[3] = {5000.0f, 5000.0f, -1000.0f};

float spheres_test[SPHERE_SIZE * spheres_test_cnt] = {
        -300.0f, 300.0f, 200.0f, 155.0f,0.45f, 0.1f, 0.45f, 0.0f, 3.0f, 3.0f, 2.0f,
        -79.0f, -10.0f, 300.0f, 100.0f, 0.1f, 0.9f, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f,
        100.0f, 50.0f, 100.0f, 70.0f, 0.1f, 0.1f, 0.9f, 0.0f, 0.0f, 0.0f, 0.0f,
        10.0f, -50.0f, 400.0f, 66.0f, 0.9f, 0.1f, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f,
        100.0f, -1000.0f, 100.0f, 500.0f, 0.1f, 0.1f, 0.9f, 0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 2000.0f, 1000.0f, 0.5f, 0.5f, 0.9f, 0.0f, 0.0f, 0.0f, 0.0f,
        1000.0f, -50.0f, 400.0f, 166.0f, 0.9f, 0.1f, 0.1f, 0.0f, 0.0f, 1.0f, 0.0f
};


void trace_ray(float *color, float *pos, float *dir, float *spheres, uint32_t spheres_count, uint32_t depth)
{
    int closest = -1;
    float dist = FLT_MAX;
    float temp;
    color[0] = color[1] = color[2] = 0.00f;
    for(int i = 0; i < spheres_count; i++)
    {
        temp = sphere_intersect(pos, dir, &spheres[i * SPHERE_SIZE], spheres[i * SPHERE_SIZE + 3]);
        if(temp < dist)
        {
            dist = temp;
            closest = i;
        }
    }
    if(closest > -1)
    {
        float temp_color[3];
        float intersect_pos[3];
        float normal[3];
        float refl[3];

        dist *= 0.995f;

        sphere_intersect_pos(intersect_pos, pos, dir, dist);
        sphere_normal(normal, intersect_pos, &spheres[closest * SPHERE_SIZE]);
        reflection(refl, dir, normal);

        if(depth < DEPTH_MAX)
        {
            trace_ray(temp_color, intersect_pos, refl, spheres, spheres_count, depth + 1);
            add(color, temp_color);

            refl[0] += 3.0f / (rand() % 1000);
            refl[1] += 3.0f / (rand() % 1000);
            trace_ray(temp_color, intersect_pos, refl, spheres, spheres_count, depth + 1);
            add(color, temp_color);

            refl[0] -= 3.0f / (rand() % 1000);
            refl[1] -= 3.0f / (rand() % 1000);
            trace_ray(temp_color, intersect_pos, refl, spheres, spheres_count, depth + 1);
            add(color, temp_color);

            refl[0] += 3.0f / (rand() % 1000);
            refl[1] -= 3.0f / (rand() % 1000);
            trace_ray(temp_color, intersect_pos, refl, spheres, spheres_count, depth + 1);
            add(color, temp_color);

            refl[0] -= 3.0f / (rand() % 1000);
            refl[1] += 3.0f / (rand() % 1000);
            trace_ray(temp_color, intersect_pos, refl, spheres, spheres_count, depth + 1);
            add(color, temp_color);
        }

        mul(color, 0.20f);
        add(color, &spheres[closest * SPHERE_SIZE + 8]);
        mul(color, 0.50f);
        float *my_color = &spheres[closest * SPHERE_SIZE + 4];
        color[0] += color[0] * my_color[0] * 0.5f;
        color[1] += color[1] * my_color[1] * 0.5f;
        color[2] += color[2] * my_color[2] * 0.5f;
        mul(color, 0.90f);

        float temp_sun[3];
        normalize(temp_sun, test_sun);
        float sun_dotp = dot(temp_sun, normal);
        if(sun_dotp > 0.0f)
        {
            color[0] += sun_dotp;
            color[1] += sun_dotp;
            color[2] += sun_dotp;
        }
    }

    color[0] += 0.05f;
    color[1] += 0.05f;
    color[2] += 0.05f;
}

void trace_rect(float *dest, int xs, int ys, int ws, int hs, int w, int h)
{
    float pos[3];
    float dir[3];

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
            trace_ray(temp_color, pos, dir, spheres_test, spheres_test_cnt, 0);
            add(color_offset, temp_color);
            trace_ray(temp_color, pos, dir, spheres_test, spheres_test_cnt, 0);
            add(color_offset, temp_color);
            mul(color_offset, 0.5f);

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
