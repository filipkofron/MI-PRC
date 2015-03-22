#ifndef SCENE_H
#define SCENE_H

#include <string>

struct scene_t
{
	float *spheres;
	int spheres_count;

	float *triangles;
	int triangles_count;

	float *light;
	int light_count;
};

extern scene_t host_scene;
extern scene_t device_scene;

void init_scene(std::string name, int width, int height);
void clean_scene();

#endif
