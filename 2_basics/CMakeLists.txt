cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(basic)

find_package(Torch REQUIRED)

add_executable(basic_multiply basic_multiply.cpp)
target_link_libraries(basic_multiply "${TORCH_LIBRARIES}")
set_property(TARGET basic_multiply PROPERTY CXX_STANDARD 14)

add_executable(basic_linear basic_linear.cpp)
target_link_libraries(basic_linear "${TORCH_LIBRARIES}")
set_property(TARGET basic_linear PROPERTY CXX_STANDARD 14)

include_directories("/workspace/libtorch-cxx11-abi-shared-with-deps-1.10.2+cu102/libtorch/include/torch/csrc/api/include/torch/")
