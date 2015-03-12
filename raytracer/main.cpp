#include <iostream>
#include <thread>
#include <unistd.h>
#include <fstream>
#include <bits/stl_bvector.h>
#include "trace.h"
#include "bmp.h"
#include "sphere.h"
#include "triangle.h"
#include "obj.h"
#include "light.h"

using namespace std;

#define TEST_WIDTH 1280
#define TEST_HEIGHT 1024
#define TEST_DIV 4

struct thread_data_t
{
    float *colors;
    int x;
    int y;
    int ws;
    int hs;
    int width;
    int height;
};

void thread_start(thread_data_t hax)
{
    trace_rect(hax.colors, hax.x, hax.y, hax.ws, hax.hs, hax.width, hax.height);
}

// start the tracing at given color array
void trace_all(int width, int height, float *colors)
{
    int ws = width / TEST_DIV;
    int hs = height / TEST_DIV;
    int threads_cnt = 0;
    std::thread **threads = new std::thread *[(TEST_DIV + 1) * (TEST_DIV + 1)];
    thread_data_t hax;
    for(uint32_t x = 0; x < width; x += ws)
    {
        for(uint32_t y = 0; y < height; y += hs)
        {
            hax.colors = colors;
            hax.x = x;
            hax.y = y;
            hax.ws = ws;
            hax.hs = hs;
            hax.width = width;
            hax.height = height;

            //trace_rect(hax.colors, hax.x, hax.y, hax.ws, hax.hs, hax.width, hax.height);

            std::thread *thread = new std::thread(thread_start, hax);
            threads[threads_cnt++] = thread;
        }
    }
    for(int i = 0; i < threads_cnt; i++)
    {
        threads[i]->join();
        delete threads[i];
    }
    delete [] threads;
}

/*
    0 - 2: pos
    4:        radius
    5 - 7:  Ambient color
    8 - 10: Diffuser color
    11 - 13: Specular color
    14:     transparency
 */

int main()
{
    int size = TEST_WIDTH * TEST_HEIGHT * 3;
    float *test = (float *) malloc(sizeof(float) * size);

    float testSpheres[SPHERE_SIZE * 3] =
            {
                    2, 0, 3, 1, 0, 0, 0, 0, 1, 0, 0.8f, 1.0f, 0.8f, 1.0f,
                    -1.5f, -1, 6, 2, 0, 0, 0, 1, 0, 0, 0.8f, 1.0f, 0.8f, 1.0f,
                    2.3, -2, 3, 1.5f, 0, 0, 0, 0, 0, 1, 0.8f, 1.0f, 0.8f, 1.0f,
            };
    float *pos, *col, *rad;

    scene.spheres = testSpheres;
    scene.spheres_count = 3;

    float testLight[LIGHT_SIZE];
    pos = LIGHT_POS(testLight);
    pos[0] = 10000;
    pos[1] = 10000;
    pos[2] = -8000;
    rad = LIGHT_RADIUS(testLight);
    *rad = 500;
    col = LIGHT_COLOR(testLight);
    col[0] = 1.0f;
    col[1] = 1.0f;
    col[2] = 0.9f;

    scene.light = testLight;
    scene.light_count = 1;

    std::ifstream objf("sample.obj");
    Obj obj(objf);
    float *trs = obj.buildTriangles(scene.triangles_count);
    scene.triangles = trs;


    std::cout << "[Prep] >> Done." << std::endl;

    trace_all(TEST_WIDTH, TEST_HEIGHT, test);
    FILE *file = fopen("/home/kofee/test.bmp", "wb+");
    srand(time(NULL));
    if(file)
    {
        write_bmp(file, test, TEST_WIDTH, TEST_HEIGHT);
        fflush(file);
        fclose(file);
    }
    else
    {
        fprintf(stderr, "File could not be opened!\n");
    }
    free(test);


    delete [] trs;
    return 0;
}
