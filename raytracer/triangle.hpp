#include <float.h>
#include "vec3.h"

float triangle_intersect(float *pos, float *dir, float *triangle, float *res_uv)
{
    float e1[3],e2[3],h[3],s[3],q[3];
    float a,f,u,v;

    sub(e1, &triangle[3], &triangle[0]);
    sub(e2, &triangle[6], &triangle[0]);

    cross(h, dir, e2);

    a = dot(e1, h);

    if(a > -0.00001f && a < 0.00001f)
        return FLT_MAX;

    f = 1.0f / a;

    sub(s, pos, &triangle[0]);

    u = f * dot(s, h);

    if(u < 0.0f || i > 1.0f)
        return FLT_MAX;

    cross(q, s, e1);
    v = f * dot(d, q);

    if(u < 0.0f || u + v > 1.0f)
        return FLT_MAX;

    t = f * dot(e2, q);

    return t;
}

void triangle_pos(float *new_pos, float *uv, float *triangle)
{
    float d = 1 - uv[0] - uv[1];
    float temp[3];
    mul(new_pos, &triangle[0], d);
    mul(temp, &triangle[3], uv[0]);
    add(new_pos, temp);
    mul(temp, &triangle[6], uv[1]);
    add(new_pos, temp);
}

void triangle_normal(float *normal, float *triangle)
{
    float e1[3];
    float e2[3];

    sub(e1, &triangle[3], &triangle[0]);
    sub(e2, &triangle[6], &triangle[0]);

    cross(normal, e1, e2);
}
