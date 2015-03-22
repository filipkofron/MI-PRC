#!/bin/bash

OBJ_OPTS="-std=c++11 --relocatable-device-code true -c"
LINK_OPTS="-std=c++11 -lcudadevrt"

rm -f raytracer.run
rm -f *.o
nvcc $OBJ_OPTS bmp.cu
nvcc $OBJ_OPTS kernel.cu
nvcc $OBJ_OPTS light.cu
nvcc $OBJ_OPTS obj.cu
nvcc $OBJ_OPTS scene.cu
nvcc $OBJ_OPTS trace.cu
nvcc $LINK_OPTS *.o -o raytracer.run
