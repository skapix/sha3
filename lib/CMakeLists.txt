project(sha3_lib C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP REQUIRED)

set(files
    util.h
    util.cpp
    common.h
    sha3_cpu.h
    sha3_cpu.cpp)

set(cu_files
    helper_cuda.h
    sha3_gpu.cu
    sha3_gpu.h)

add_library(${PROJECT_NAME} STATIC ${files} ${cu_files})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})


if (OpenMP_FOUND)
    target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
