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

void triangle_test()
{
    float pos[3] = {0.0f, 0.0f, -1.0f};
    float dir[3] = {0.0f, -1.0f, -1.0f};
    float res_uv[3] = {0.0f, 0.0f, 0.0f};
    float res_pos[3] = {0.0f, 0.0f, 0.0f};

    normalize(dir);

    float triangle[TRIANGLE_SIZE] = {
            -1.0f, 1.0f, 0.0f,
            1.0f, 1.0f, 0.0f,
            0.0f, -1.0f, 0.0f,

            666.0f, 666.0f, 666.0f, 666.0f, 666.0f, 666.0f, 666.0f, 666.0f, 666.0f, 666.0f};
    float res = triangle_intersect(pos, dir, triangle, res_uv);
    triangle_pos(res_pos, res_uv, triangle);
    std::cout << "result distance: " << res << std::endl;
    std::cout << "result uv: " << res_uv[0] << ", " << res_uv[1] << std::endl;
    std::cout << "result pos: " << res_pos[0] << ", " << res_pos[1] << ", " << res_pos[2] << std::endl;
}

int main()
{/*
    int size = TEST_WIDTH * TEST_HEIGHT * 3;
    float *test = (float *) malloc(sizeof(float) * size);

    init_scene("sample", TEST_WIDTH, TEST_HEIGHT);

    std::cout << "[Prep] >> Done." << std::endl;

    trace_all(TEST_WIDTH, TEST_HEIGHT, test);
    clean_scene();


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
    free(test);*/



    triangle_test();
    return 0;
}
