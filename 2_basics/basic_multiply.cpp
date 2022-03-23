#include <torch/torch.h>
#include <torch/script.h>

#include <iostream>
#include <iomanip>

void print_tensor_size(const torch::Tensor&);
void print_script_module(const torch::jit::script::Module& module, size_t spaces = 0);

int main() {
    // cmake -DCMAKE_PREFIX_PATH=/workspace/libtorch-cxx11-abi-shared-with-deps-1.10.2+cu102/libtorch ..

    torch::cuda::is_available();
    // Create tensor
    torch::Tensor x = torch::tensor(1.0, torch::requires_grad());
    torch::Tensor w = torch::tensor(2.0, torch::requires_grad());
    torch::Tensor b = torch::tensor(3.0, torch::requires_grad());

    // Build computational graph 
    auto y = w * x + b;

    // Compute the gradients
    y.backward();

    // Print the gradients
    std::cout << x.grad() << '\n';  // x.grad() = 2
    std::cout << w.grad() << '\n';  // w.grad() = 1
    std::cout << b.grad() << "\n\n";  // b.grad() = 1

    return 0;
}
