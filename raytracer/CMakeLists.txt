cmake_minimum_required(VERSION 2.8.4)
project(raytracer)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -O2 -std=c++11 -pthread")

configure_file(sample.obj sample.obj COPYONLY)
configure_file(sample.mtl sample.mtl COPYONLY)
configure_file(sample.sph sample.sph COPYONLY)
configure_file(sample.lit sample.lit COPYONLY)


set(SOURCE_FILES    bmp.h
                    bmp.cpp
                    light.h
                    light.cpp
                    main.cpp
                    obj.cpp
                    obj.h
                    rand.h
                    scene.cpp
                    scene.h
                    sphere.h
                    trace.h
                    trace.cpp
                    triangle.h
                    vec3.h)
add_executable(raytracer ${SOURCE_FILES})