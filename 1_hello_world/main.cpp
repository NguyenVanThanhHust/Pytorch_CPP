#include <torch/torch.h>
#include <iostream>

int main() {
  // cmake -DCMAKE_PREFIX_PATH=/workspace/libtorch-cxx11-abi-shared-with-deps-1.10.2+cu102/libtorch ..
  torch::Tensor tensor = torch::eye(3);
  std::cout << tensor << std::endl;
}
