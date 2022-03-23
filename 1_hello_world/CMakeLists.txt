cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main "${TORCH_LIBRARIES}")
include_directories("/workspace/libtorch-cxx11-abi-shared-with-deps-1.10.2+cu102/libtorch/include/torch/csrc/api/include/torch/")
set_property(TARGET main PROPERTY CXX_STANDARD 14)
