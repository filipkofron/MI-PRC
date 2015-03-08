cmake_minimum_required(VERSION 2.8.4)
project(raytracer)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

set(SOURCE_FILES    bmp.h
                    bmp.cpp
                    main.cpp
                    obj.cpp
                    obj.hpp
                    rand.hpp
                    trace.h
                    trace.cpp
                    vec3.h)
add_executable(raytracer ${SOURCE_FILES})