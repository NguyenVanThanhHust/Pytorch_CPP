cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(linear_regression)

find_package(Torch REQUIRED)

add_executable(linear_regression main.cpp)
target_link_libraries(linear_regression "${TORCH_LIBRARIES}")
set_property(TARGET linear_regression PROPERTY CXX_STANDARD 14)

include_directories("/workspace/libtorch-cxx11-abi-shared-with-deps-1.10.2+cu102/libtorch/include/torch/csrc/api/include/torch/")
