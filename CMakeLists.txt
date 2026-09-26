cmake_minimum_required(VERSION 3.30)
project(tinyinfer C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

include_directories(include)

add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wsign-conversion
        -Wno-sign-conversion
)

add_compile_definitions(_POSIX_C_SOURCE=200112L)

set(CMAKE_C_FLAGS_RELEASE "-O2 -ffunction-sections -fdata-sections -DNDEBUG")
set(CMAKE_C_FLAGS_DEBUG   "-g -O0 -fsanitize=address,undefined")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-Wl,--gc-sections")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG   "-fsanitize=address,undefined")

link_libraries(m)

option(USE_ESP_DSP "Link ESP-DSP library for optimized vector operations" OFF)
if(USE_ESP_DSP)
    add_compile_definitions(TI_USE_ESP_DSP=1)
endif()


add_executable(tinyinfer      main.c src/tensor.c src/layers.c src/activations.c)
add_executable(test_tensor    tests/test_tensor.c src/tensor.c)
add_executable(test_layer     tests/test_layer.c src/tensor.c src/layers.c)
add_executable(test_layer_int8 tests/test_layer_int8.c src/tensor.c src/layers.c)
add_executable(test_activations tests/test_activations.c src/tensor.c src/activations.c)