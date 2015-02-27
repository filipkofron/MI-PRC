#include "trace.h"
#include "circle.h"

float center[3] = {0.0f, 100.0f, 200.0f};
float r = 100.0f;

void trace_ray(float *color, float *pos, float *dir)
{
    float dist =  sphere_intersect(pos, dir, center, r) / 100.0f;
    if(dist > 0.001)
    {
        color[0] = color[1] = color[2] = 1.0f - dist;
    }
    else color[0] = color[1] = color[2] = 0.0f;
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
            init_vec3(dir, pos[0] * 0.001f, pos[1] * 0.001f, 1.0f);
            normalize(dir);
            trace_ray(&dest[(y * w + x) * 3], pos, dir);
        }
    }
}
