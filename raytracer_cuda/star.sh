#!/bin/bash

rm -f raytracer.run
rm -f *.o
nvcc -std=c++11 -c bmp.cu
nvcc -std=c++11 -c kernel.cu
nvcc -std=c++11 -c light.cu
nvcc -std=c++11 -c obj.cu
nvcc -std=c++11 -c scene.cu
nvcc -std=c++11 -c trace.cu
nvcc -std=c++11 *.o -o raytracer.run
