#include "light.h"
#include "trace.h"

// simple max that will not evaluate twice
static float my_max(float a, float b)
{
    return a > b ? a : b;
}

/*
calculate light at a given position of ray intersect
*/
void calc_light(float *ray_pos, float *obj_normal, float *light_res, scene_t *scene, color_t *colors)
{
    float dir_to_light[3];
    float dir_to_light_norm[3];
    float incom_light[3];
    float incom_light_norm[3];
    float refl[3];

    float dummy[3];

    float diffuse[3];
    float specular[3];

    light_res[0] = light_res[1] = light_res[2] = 0.0f;
    set_vec3(diffuse, light_res);
    set_vec3(specular, light_res);

    for(int i = 0; i < scene->light_count; i++)
    {
        float *curr_light = LIGHT_INDEX(i, scene->light);
        sub(dir_to_light, LIGHT_POS(curr_light), ray_pos);
        normalize(dir_to_light_norm, dir_to_light);
        float cos = dot(obj_normal, dir_to_light_norm);
        cos = cos < 0.0f ? 0.0f : cos;
        mul(diffuse, colors->diffuse, cos);


        sub(incom_light, ray_pos, LIGHT_POS(curr_light));
        normalize(incom_light_norm, incom_light);
        float dots = -dot(incom_light_norm, obj_normal);
        float lens = 2.0f * dots;
        mul(refl, obj_normal, lens);
        add(refl, incom_light_norm);
        normalize(refl);
        float spec = my_max(-dot(refl, incom_light_norm), 0);
        spec = powf(spec, 20.0f);
        mul(specular, colors->specular, spec);

        color_t dummyCol;
        /*if(find_intersect(ray_pos, dir_to_light_norm, dummy, dummy, dummy, &dummyCol))
        {
            set_vec3(diffuse, light_res);
            set_vec3(specular, light_res);
        }*/
    }

    add(light_res, diffuse);
    add(light_res, specular);
}
