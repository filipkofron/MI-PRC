#include "scene.h"

static void load_spheres(std::ifstream &ifs)
{
		std::vector<float> sph;
    while(!ifs.fail())
    {
        std::string line;
        std::getline(ifs, line);
        std::stringstream ss;
        ss << line;
        
        float a,b,c,r;
        ss >> a >> b >> c >> r;
        
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
        ss >> l[0] >> l[1] >> l[2];
        sph.push_back(a);
    }
    scene.spheres_count = sph.size() / SPHERE_SIZE;
    scene.spheres = new float[sph.size()];
    
    memcpy(scene.spheres, sph.data(), sizeof(float) * sph.size());
}


static void load_lights(std::ifstream &ifs)
{
		std::vector<float> lit;
    while(!ifs.fail())
    {
        std::string line;
        std::getline(ifs, line);
        std::stringstream ss;
        ss << line;
        
        float a,b,c,r;
        ss >> a >> b >> c >> r;
        
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
    scene.lights_count = lkt.size() / SPHERE_SIZE;
    scene.lights = new float[lit.size()];
    
    memcpy(scene.spheres, lit.data(), sizeof(float) * lit.size());
}


void init_scene(std::string name, int width, int height)
{
	  std::ifstream spheres(name + ".sph");
	  
	  spheres.close();
	  
	  std::ifstream lights(name + ".lit");
	  lights.close();
}

void clean_scene()
{
	delete [] scene.triangles;
	delete [] scene.spheres;
	delete [] scene.lights;
}

