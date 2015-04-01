#!/bin/bash

source /etc/profile
source /etc/profile.env

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:$PATH

cd /home/kofrofil/MI-PRC/raytracer_cuda/

# OBJ_OPTS="-Xptxas -v -maxrregcount 28 -std=c++11 --relocatable-device-code true -c"
OBJ_OPTS="-maxrregcount 28 -std=c++11 -gencode=arch=compute_20,code=sm_20 -Xptxas -v -gencode=arch=compute_20,code=compute_20 --relocatable-device-code true -c"
LINK_OPTS=" -std=c++11 -lcudadevrt"

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

echo "-> Build done, running raytracer.."

rm -f test.bmp
./raytracer.run

echo "-> All done."
