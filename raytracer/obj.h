#ifndef OBJ_HPP
#define OBJ_HPP

#include <vector>
#include <map>
#include "vec3.h"

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
        std::vector<vec3_t> vecs;
        std::vector<Face> faces;
    };
public:
    Obj(std::ifstream &ifs);
    void loadFile(std::ifstream &ifs);
    std::map<std::string, Object> objects;
};

#endif
