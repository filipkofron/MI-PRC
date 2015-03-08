#include <iostream>
#include <thread>
#include <unistd.h>
#include <fstream>
#include <bits/stl_bvector.h>
#include "trace.h"
#include "bmp.h"
#include "sphere.h"
#include "triangle.hpp"
#include "obj.hpp"

using namespace std;

#define TEST_WIDTH 1280
#define TEST_HEIGHT 1024
#define TEST_DIV 8

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

int main()
{
    int size = TEST_WIDTH * TEST_HEIGHT * 3;
    float *test = (float *) malloc(sizeof(float) * size);

    float testSphere[SPHERE_SIZE];
    float *pos = SPHERE_POS(testSphere);
    pos[0] = 0;
    pos[1] = 0;
    pos[2] = 1000;
    float *rad = SPHERE_RADIUS(testSphere);
    *rad = 400;

    float testTriangle[TRIANGLE_SIZE];
    pos = TRIANGLE_POS(testTriangle);
    pos[0] = -100;
    pos[1] = -100;
    pos[2] = 100;

    pos[3] = 100;
    pos[4] = -100;
    pos[5] = 200;

    pos[6] = 100;
    pos[7] = 100;
    pos[8] = 50;

    scene.spheres = testSphere;
    scene.spheres_count = 0;
    scene.triangles = testTriangle;
    scene.triangles_count = 1;


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

    std::ifstream objf("/home/kofee/test.obj");
    Obj obj(objf);
    std::cout << "faces: " << obj.faces.size() << " vecs: " << obj.vecs.size() << std::endl;
    return 0;
}
