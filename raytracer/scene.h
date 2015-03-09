#ifndef SCENE_H
#define SCENE_H

struct scene_t
{
    float *spheres;
    int spheres_count;

    float *triangles;
    int triangles_count;

    float *light;
    int light_count;
};

#endif
