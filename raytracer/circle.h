#ifndef CIRCLE_H
#define CIRCLE_H

float sphere_intersect(float *pos, float *dir, float *center, float r)
{
    float ray_pos_x = pos[0] - center[0];
    float ray_pos_y = pos[1] - center[1];
    float ray_pos_z = pos[2] - center[2];

    float b = ray_pos_x * dir[0] + ray_pos_y * dir[1] + ray_pos_z * dir[2];
    float t = b * b - ray_pos_x * ray_pos_x - ray_pos_y * ray_pos_y - ray_pos_z * ray_pos_z + r * r;

    float sol;

    if(t < 0)
        return 0.0f;

    sol = -b - sqrt(t);

    if(sol > 0)
        return sol;

    sol = -b + sqrt(t);

    if(sol > 0)
        return sol;

    return 0.0f;
}

#endif
