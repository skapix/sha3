cmake_minimum_required(VERSION 3.0)

set(CMAKE_CUDA_FLAGS "-gencode arch=compute_50,code=sm_50")

project (
    sha3
    VERSION 1.0
    DESCRIPTION "Sha3"
    LANGUAGES C CXX CUDA
)

# Set up CLI11.
add_subdirectory (external/CLI11 EXCLUDE_FROM_ALL)

# Generate projects.
add_subdirectory(lib)
add_subdirectory(sha3)
add_subdirectory(sha3_batch)
add_subdirectory(test)
add_subdirectory(benchmark)
