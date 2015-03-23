#!/bin/bash

OBJ_OPTS="-std=c++11 --relocatable-device-code true -c"
LINK_OPTS="-std=c++11 -lcudadevrt"

echo "-> Cleaning previous run."
rm -f raytracer.run
rm -f *.o

echo "-> Building bmp.cu"
nvcc $OBJ_OPTS bmp.cu

echo "-> Building kernel.cu"
nvcc $OBJ_OPTS kernel.cu

echo "-> Building light.cu"
nvcc $OBJ_OPTS light.cu

echo "-> Building obj.cu"
nvcc $OBJ_OPTS obj.cu

echo "-> Building scene.cu"
nvcc $OBJ_OPTS scene.cu

echo "-> Building trace.cu"
nvcc $OBJ_OPTS trace.cu

echo "-> Linking raytracer.run"
nvcc $LINK_OPTS *.o -o raytracer.run

echo "-> Done!"
