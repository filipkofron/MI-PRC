#ifndef OBJ_HPP
#define OBJ_HPP

#include <vector>
#include <map>
#include "vec3.h"

/* *.obj format structured class for loading and processing it */
class Obj
{
private:
	struct Material
	{
		float ambient[3];
		float diffuse[3];
		float specular[3];
		float specular_exponent;
		float transparency;
	};
	struct Face
	{
		std::vector<int> vecs;
	};
	struct Object
	{
		std::string material;
		std::vector<vec3_t> vecs;
		std::vector<Face> faces;
	};
public:
	// will construct the array from object
	Obj(std::ifstream &ifs);
	void loadObj(std::ifstream &ifs);
	void loadMtl(std::ifstream &ifs);
	std::map<std::string, Object> objects;
	std::map<std::string, Material> materials;

	// allocate and build triangle array from the loaded obj instance
	float *buildTriangles(int &size);
};

#endif
