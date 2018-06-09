project(sha3_test)

set(CMAKE_CXX_STANDARD 17)

find_package(GTest)
if(NOT GTEST_FOUND)
    message(WARNING "Gtest not found, tests can't be built.")
    return()
endif()

enable_testing()

add_executable(${PROJECT_NAME} test_sha3.cpp)
target_link_libraries(${PROJECT_NAME} sha3_lib GTest::GTest GTest::Main)
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_test(AllTests ${PROJECT_NAME})
