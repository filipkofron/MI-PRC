#include "scene.cuh"
#include "sphere.cuh"
#include "obj.cuh"
#include "light.cuh"
#include "triangle.cuh"
#include "common.cuh"
#include <vector>
#include <fstream>
#include <sstream>
#include <cstring>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// scene instance
scene_t host_scene;
scene_t dev_scene;


// Will load spheres from given file
static void load_spheres(std::ifstream &ifs)
{
	std::cout << "[Prep] >> Loading spheres." << std::endl;
	std::vector<float> sph;
	while (!ifs.fail())
	{
		std::string line;
		std::getline(ifs, line);
		std::stringstream ss;
		ss << line;

		float a, b, c, r;
		ss >> a >> b >> c >> r;

		if (ss.fail())
		{
			break;
		}

		sph.push_back(a);
		sph.push_back(b);
		sph.push_back(c);
		sph.push_back(r);

		float l[3];
		ss >> l[0] >> l[1] >> l[2];
		sph.push_back(l[0]);
		sph.push_back(l[1]);
		sph.push_back(l[2]);
		ss >> l[0] >> l[1] >> l[2];
		sph.push_back(l[0]);
		sph.push_back(l[1]);
		sph.push_back(l[2]);
		ss >> l[0] >> l[1] >> l[2];
		sph.push_back(l[0]);
		sph.push_back(l[1]);
		sph.push_back(l[2]);
		ss >> a;
		sph.push_back(a);
	}

	host_scene.spheres_count = (int)sph.size() / SPHERE_SIZE;
	host_scene.spheres = new float[sph.size()];

	memcpy(host_scene.spheres, sph.data(), sizeof(float)* sph.size());
}

// will load lights
static void load_lights(std::ifstream &ifs)
{
	std::cout << "[Prep] >> Loading lights." << std::endl;
	std::vector<float> lit;
	while (!ifs.fail())
	{
		std::string line;
		std::getline(ifs, line);
		std::stringstream ss;
		ss << line;

		float a, b, c, r;
		ss >> a >> b >> c >> r;

		if (ss.fail())
		{
			break;
		}

		lit.push_back(a);
		lit.push_back(b);
		lit.push_back(c);
		lit.push_back(r);

		float l[3];
		ss >> l[0] >> l[1] >> l[2];
		lit.push_back(l[0]);
		lit.push_back(l[1]);
		lit.push_back(l[2]);
	}
	host_scene.light_count = (int)lit.size() / LIGHT_SIZE;
	host_scene.light = new float[lit.size()];

	memcpy(host_scene.light, lit.data(), sizeof(float)* lit.size());
}

// will load triangles
void load_triangles(std::ifstream &ifs)
{
	Obj obj(ifs);
	float *trs = obj.buildTriangles(host_scene.triangles_count);
	host_scene.triangles = trs;
}

// initialize whole scene
void init_scene(std::string name, int width, int height)
{
	std::ifstream objf(name + ".obj");
	load_triangles(objf);
	objf.close();

	std::ifstream spheres(name + ".sph");
	load_spheres(spheres);
	spheres.close();

	std::ifstream lights(name + ".lit");
	load_lights(lights);
	lights.close();

	dev_scene.light_count = host_scene.light_count;
	dev_scene.spheres_count = host_scene.spheres_count;
	dev_scene.triangles_count = host_scene.triangles_count;

	cudaSafeMalloc((void **) &dev_scene.light, dev_scene.light_count * sizeof(float)* LIGHT_SIZE);
	cudaMemcpy(dev_scene.light, host_scene.light, dev_scene.light_count * sizeof(float)* LIGHT_SIZE, cudaMemcpyHostToDevice);

	cudaSafeMalloc((void **) &dev_scene.spheres, dev_scene.spheres_count * sizeof(float)* SPHERE_SIZE);
	cudaMemcpy(dev_scene.spheres, host_scene.spheres, dev_scene.spheres_count * sizeof(float)* SPHERE_SIZE, cudaMemcpyHostToDevice);

	cudaSafeMalloc((void **) &dev_scene.triangles, dev_scene.triangles_count * sizeof(float)* TRIANGLE_SIZE);
	cudaMemcpy(dev_scene.triangles, host_scene.triangles, dev_scene.triangles_count * sizeof(float)* TRIANGLE_SIZE, cudaMemcpyHostToDevice);
}

// cleanup the scene
void clean_scene()
{
	delete[] host_scene.triangles;
	delete[] host_scene.spheres;
	delete[] host_scene.light;

	cudaSafeFree(dev_scene.light);
	cudaSafeFree(dev_scene.spheres);
	cudaSafeFree(dev_scene.triangles);
}
