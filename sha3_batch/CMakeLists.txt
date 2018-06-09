project(sha3_batch C CXX CUDA)

find_package(OpenMP)
if (NOT OpenMP_FOUND)
    message(WARNING "OpenMP not found, sha3_batch can't be built.")
    return()
endif()

set(CMAKE_CXX_STANDARD 17)


add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} sha3_lib CLI11)
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
