cmake_minimum_required(VERSION 2.8.4)
project(raytracer)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

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