project(teb)
cmake_minimum_required(VERSION 3.16)
find_package(Eigen3)
add_subdirectory(third_party/ifopt)
add_executable(test src/test.cpp)
target_include_directories(test SYSTEM
                           PRIVATE third_party/ifopt/ifopt_core/include
                                   third_party/ifopt/ifopt_ipopt/include
                                   ${EIGEN3_INCLUDE_DIR})
target_link_libraries(test PRIVATE ifopt_core ifopt_ipopt)
