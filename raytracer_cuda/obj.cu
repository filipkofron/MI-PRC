#include <fstream>
#include <sstream>
#include <iostream>
#include <string.h>
#include "obj.cuh"
#include "triangle.cuh"

Obj::Obj(std::ifstream &ifs)
{
	loadObj(ifs);
}

void Obj::loadObj(std::ifstream &ifs)
{
	Object *obj = NULL;
	int vec_offset = 0;
	int vec_acc = 0;
	std::string name = "unknown";
	while (!ifs.fail())
	{
		std::string line;
		std::getline(ifs, line);
		std::stringstream ss;
		ss << line;

		std::string tag;
		ss >> tag;
		if (tag == "mtllib")
		{
			std::string filename;
			ss >> filename;
			std::ifstream mtlifs(filename);
			loadMtl(mtlifs);
			mtlifs.close();
		}
		if (tag == "usemtl")
		{
			ss >> obj->material;
		}
		if (tag == "o")
		{
			if (obj)
			{
				objects[name] = *obj;
				delete obj;
			}
			ss >> name;
			obj = new Object;
			std::cout << "[Prep] >> Loading new object: " << name << std::endl; \
				vec_offset += vec_acc;
			vec_acc = 0;
		}
		if (tag == "v")
		{
			vec3_t vec;
			ss >> vec.x >> vec.y >> vec.z;
			obj->vecs.push_back(vec);
			vec_acc++;
		}
		if (tag == "f")
		{
			Face face;
			while (!ss.fail())
			{
				int fi;
				ss >> fi;
				if (ss.fail())
					break;
				face.vecs.push_back(fi - vec_offset);
			}
			obj->faces.push_back(face);
		}
	}
	if (obj)
	{
		objects[name] = *obj;
		delete obj;
	}
}

void Obj::loadMtl(std::ifstream &ifs)
{
	Material *mat = NULL;
	std::string name = "unknown";

	while (!ifs.fail())
	{
		std::string line;
		std::getline(ifs, line);
		std::stringstream ss;
		ss << line;

		std::string tag;
		ss >> tag;
		if (tag == "newmtl")
		{
			if (mat)
			{
				materials[name] = *mat;
				delete mat;
			}
			ss >> name;
			mat = new Material;
			std::cout << "[Prep] >> Loading new material: " << name << std::endl;
		}
		if (tag == "ns")
		{
			float ns;
			ss >> ns;
			mat->specular_exponent = ns;
		}
		if (tag == "Ka")
		{
			vec3_t vec;
			ss >> vec.x >> vec.y >> vec.z;
			set_vec3(mat->ambient, vec.arr);
		}
		if (tag == "Kd")
		{
			vec3_t vec;
			ss >> vec.x >> vec.y >> vec.z;
			set_vec3(mat->diffuse, vec.arr);
		}
		if (tag == "Ks")
		{
			vec3_t vec;
			ss >> vec.x >> vec.y >> vec.z;
			set_vec3(mat->specular, vec.arr);
		}
		if (tag == "d" || tag == "Tr")
		{
			float tr;
			ss >> tr;
			mat->transparency = tr;
		}
	}
	if (mat)
	{
		materials[name] = *mat;
		delete mat;
	}
}

#define ADD_VEC3(trs, vec3) \
{\
	trs.push_back(vec3.x); \
	trs.push_back(vec3.y); \
	trs.push_back(vec3.z); \
}

#define ADD_ARR3(trs, vec3) \
{\
	trs.push_back(vec3[0]); \
	trs.push_back(vec3[1]); \
	trs.push_back(vec3[2]); \
}

float *Obj::buildTriangles(int &size)
{
	/*
	0 - 2: pos a
	3 - 5: pos b
	6 - 8: pos c
	9 - 11:  Ambient color
	12 - 14: Diffuser color
	15 - 17: Specular color
	18:     transparency
	*/
	std::vector<float> trs;
	for (auto pair : objects)
	{
		Object obj = pair.second;
		Material mat = materials[obj.material];

		for (Face face : obj.faces)
		{
			vec3_t v3 = obj.vecs[face.vecs[0] - 1];
			ADD_VEC3(trs, v3);
			v3 = obj.vecs[face.vecs[1] - 1];
			ADD_VEC3(trs, v3);
			v3 = obj.vecs[face.vecs[2] - 1];
			ADD_VEC3(trs, v3);

			ADD_ARR3(trs, mat.ambient);
			ADD_ARR3(trs, mat.diffuse);
			ADD_ARR3(trs, mat.specular);
			trs.push_back(mat.transparency);

			if (face.vecs.size() == 4)
			{
				v3 = obj.vecs[face.vecs[2] - 1];
				ADD_VEC3(trs, v3);
				v3 = obj.vecs[face.vecs[3] - 1];
				ADD_VEC3(trs, v3);
				v3 = obj.vecs[face.vecs[0] - 1];
				ADD_VEC3(trs, v3);

				ADD_ARR3(trs, mat.ambient);
				ADD_ARR3(trs, mat.diffuse);
				ADD_ARR3(trs, mat.specular);
				trs.push_back(mat.transparency);
			}
		}
	}

	float *res = new float[trs.size()];
	memcpy(res, trs.data(), sizeof(float)* trs.size());
	size = (int)trs.size() / TRIANGLE_SIZE;
	return res;
}
