#!/bin/bash

rm -f raytracer.run
rm -f *.o
nvcc -c bmp.cu
nvcc -c kernel.cu
nvcc -c light.cu
nvcc -c obj.cu
nvcc -c scene.cu
nvcc -c trace.cu
nvcc *.o -o raytracer.run
