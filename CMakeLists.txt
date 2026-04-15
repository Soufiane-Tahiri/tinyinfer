cmake_minimum_required(VERSION 3.30)
project(tinyinfer C)

set(CMAKE_C_STANDARD 99)

include_directories(include)

# ============================================
# Optimization flags: Maximize performance without sacrificing accuracy
# ============================================
# -O2: Good optimization without -march-native (which causes Win GCC issues)
# -ffunction-sections -fdata-sections: Enable linker garbage collection
# -Wl,--gc-sections: Remove unused sections
# CRITICAL: NO -ffast-math or -march=native on Windows GCC (causes assembler errors)
set(CMAKE_C_FLAGS_RELEASE "-O2 -ffunction-sections -fdata-sections -DNDEBUG")
set(CMAKE_C_LINK_FLAGS_RELEASE "-Wl,--gc-sections")
set(CMAKE_C_FLAGS_DEBUG "-g -O0")

# Link math library for activations (expf, fmaxf, llrintf)
link_libraries(m)

# Optional ESP-DSP integration (set -DUSE_ESP_DSP=ON for ESP32 builds)
option(USE_ESP_DSP "Link ESP-DSP library for optimized vector operations" OFF)

if(USE_ESP_DSP)
    # ESP-DSP provides SIMD-optimized kernels for ESP32 (requires ESP-IDF)
    add_compile_definitions(TI_USE_ESP_DSP=1)
endif()

# ============================================
# Executables
# ============================================
add_executable(tinyinfer main.c src/tensor.c src/layers.c src/activations.c)
add_executable(test_tensor tests/test_tensor.c src/tensor.c)
add_executable(test_layer tests/test_layer.c src/tensor.c src/layers.c)
add_executable(test_layer_int8 tests/test_layer_int8.c src/tensor.c src/layers.c)
add_executable(test_activations tests/test_activations.c src/tensor.c src/activations.c)


