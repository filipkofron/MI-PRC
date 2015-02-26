#include <iostream>
#include "trace.h"
#include "bmp.h"

using namespace std;

#define TEST_WIDTH 512
#define TEST_HEIGHT 512
#define TEST_DIV 8

void trace_all(int width, int height, float *colors)
{
    int ws = width / TEST_DIV;
    int hs = height / TEST_DIV;
    for(uint32_t x = 0; x < width; x += ws)
    {
        for(uint32_t y = 0; y < height; y += hs)
        {
            trace_rect(colors, x, y, ws, hs, width, height);
        }
    }
}

int main()
{
    int size = TEST_WIDTH * TEST_HEIGHT * 3;
    float *test = (float *) malloc(sizeof(float) * size);
    trace_all(TEST_WIDTH, TEST_HEIGHT, test);
    FILE *file = fopen("/tmp/test.bmp", "wb+");
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
    return 0;
}
