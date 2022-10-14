cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_library(utils utils.cpp image_components.cpp)
