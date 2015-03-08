#ifndef OBJ_HPP
#define OBJ_HPP

#include <vector>
#include "vec3.h"

class Obj
{
private:
    struct Face
    {
        std::vector<int> vecs;
    };
public:
    Obj(std::ifstream &ifs);
    void loadFile(std::ifstream &ifs);
    std::vector<vec3_t> vecs;
    std::vector<Face> faces;
};

#endif
