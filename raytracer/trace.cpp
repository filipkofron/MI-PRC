#include "trace.h"

void trace_ray(float *color, float *pos, float *dir)
{
    cross(color, pos, dir);
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
            init_vec3(dir, 0.0f, 0.0f, 1.0f);
            trace_ray(&dest[(y * w + x) * 3], pos, dir);
        }
    }
}
