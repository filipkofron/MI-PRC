#!/bin/bash

rm -rf raytracer.run
nvcc *.cu -o raytracer.run
