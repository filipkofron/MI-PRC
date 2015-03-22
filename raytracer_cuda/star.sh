#!/bin/bash

rm -f raytracer.run
rm -f *.o
nvcc -std=c++0x -c bmp.cu
nvcc -std=c++0x -c kernel.cu
nvcc -std=c++0x -c light.cu
nvcc -std=c++0x -c obj.cu
nvcc -std=c++0x -c scene.cu
nvcc -std=c++0x -c trace.cu
nvcc -std=c++0x *.o -o raytracer.run
