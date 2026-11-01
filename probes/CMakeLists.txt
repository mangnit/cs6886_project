# Builds the patched llama.cpp b10434 (ggml + llama libraries only) and the probes.
#
#   cmake -S probes -B build/probes-sm61 -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/probes-sm61 -j3
#
# The CUDA flags are llama.cpp's own (-use_fast_math, set inside ggml-cuda); nothing
# here overrides them, so the kernels under test are compiled exactly as upstream
# compiles them.
cmake_minimum_required(VERSION 3.18)
project(cs6886_probes LANGUAGES C CXX CUDA)

set(LLAMA_CPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/llama.cpp-b10434)

set(GGML_CUDA            ON  CACHE BOOL "" FORCE)
set(GGML_NATIVE          OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_UI       OFF CACHE BOOL "" FORCE)
set(LLAMA_USE_PREBUILT_UI OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "")   # ON for the Kaggle bundle: one libggml-cuda.so, small tools
# llama-server (experiment 4g: the prior work's cache_prompt workload). Only targets named on the
# command line are built (EXCLUDE_FROM_ALL below); no HTTPS, so no OpenSSL dependency on Kaggle.
set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   ON  CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL        OFF CACHE BOOL "" FORCE)

add_subdirectory(${LLAMA_CPP_DIR} llama.cpp EXCLUDE_FROM_ALL)
find_package(CUDAToolkit REQUIRED)

foreach(tool matmul_paths gate_probe)
    add_executable(${tool} ${tool}.cpp)
    target_link_libraries(${tool} PRIVATE ggml CUDA::cudart)
    target_compile_features(${tool} PRIVATE cxx_std_17)
endforeach()

foreach(tool kv_paths quantize_pure ppl_paths kv_bisect real_replay)
    add_executable(${tool} ${tool}.cpp)
    target_link_libraries(${tool} PRIVATE llama ggml CUDA::cudart)
    target_compile_features(${tool} PRIVATE cxx_std_17)
endforeach()
