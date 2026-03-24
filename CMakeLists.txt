cmake_minimum_required(VERSION 3.30)
project(tinyinfer C)

set(CMAKE_C_STANDARD 99)

include_directories(include)

add_executable(tinyinfer main.c src/tensor.c src/layers.c)
add_executable(test_tensor tests/test_tensor.c src/tensor.c)
add_executable(test_layer tests/test_layer.c src/tensor.c src/layers.c)