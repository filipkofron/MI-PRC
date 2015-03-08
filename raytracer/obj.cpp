#include <fstream>
#include <sstream>
#include "obj.hpp"

Obj::Obj(std::ifstream &ifs)
{
    loadFile(ifs);
}

void Obj::loadFile(std::ifstream &ifs)
{
    vecs.clear();
    faces.clear();
    while(!ifs.fail())
    {
        std::string line;
        std::getline(ifs, line);
        std::stringstream ss;
        ss << line;

        std::string tag;
        ss >> tag;
        if(tag == "v")
        {
            vec3_t vec;
            ss >> vec.x >> vec.y >> vec.z;
            vecs.push_back(vec);
        }
        if(tag == "f")
        {
            Face face;
            while(!ss.fail())
            {
                int fi;
                ss >> fi;
                face.vecs.push_back(fi);
            }
            faces.push_back(face);
        }
    }
}
