#include "scene.h"
#include "sphere.h"
#include "obj.h"
#include "light.h"
#include <vector>
#include <fstream>
#include <sstream>
#include <cstring>
#include <iostream>

// scene instance
// TODO: use ptrs and have 2 scenes at various memories
scene_t scene;


// Will load spheres from given file
static void load_spheres(std::ifstream &ifs)
{
    std::cout << "[Prep] >> Loading spheres." << std::endl;
	std::vector<float> sph;
    while(!ifs.fail())
    {
        std::string line;
        std::getline(ifs, line);
        std::stringstream ss;
        ss << line;
        
        float a,b,c,r;
        ss >> a >> b >> c >> r;

        if(ss.fail())
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

    scene.spheres_count = (int) sph.size() / SPHERE_SIZE;
    scene.spheres = new float[sph.size()];

    memcpy(scene.spheres, sph.data(), sizeof(float) * sph.size());
}

// will load lights
static void load_lights(std::ifstream &ifs)
{
    std::cout << "[Prep] >> Loading lights." << std::endl;
    std::vector<float> lit;
    while(!ifs.fail())
    {
        std::string line;
        std::getline(ifs, line);
        std::stringstream ss;
        ss << line;
        
        float a,b,c,r;
        ss >> a >> b >> c >> r;

        if(ss.fail())
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
    scene.light_count = (int) lit.size() / LIGHT_SIZE;
    scene.light = new float[lit.size()];
    
    memcpy(scene.light, lit.data(), sizeof(float) * lit.size());
}

// will load triangles
void load_triangles(std::ifstream &ifs)
{
    Obj obj(ifs);
    float *trs = obj.buildTriangles(scene.triangles_count);
    scene.triangles = trs;
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
}

// cleanup the scene
void clean_scene()
{
	delete [] scene.triangles;
	delete [] scene.spheres;
	delete [] scene.light;
}

