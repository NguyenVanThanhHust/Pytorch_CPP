cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(convolutional_neural_network)

find_package(Torch REQUIRED)

add_executable(convolutional_neural_network src/main.cpp
                                            src/convnet.cpp
                                            src/imagefolder_dataset.cpp
                                            include/convnet.h
                                            include/imagefolder_dataset.h
              )
target_link_libraries(convolutional_neural_network "${TORCH_LIBRARIES}")
set_property(TARGET convolutional_neural_network PROPERTY CXX_STANDARD 14)

include_directories("/workspace/libtorch-cxx11-abi-shared-with-deps-1.10.2+cu102/libtorch/include/torch/csrc/api/include/torch/")
