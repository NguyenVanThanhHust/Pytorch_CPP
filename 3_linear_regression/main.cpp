#include <torch/torch.h>
#include <iostream>
#include <iomanip>

int main(){
    std::cout << "Linear regression \n ";
    std::cout << "Training on CPU \n";

    // Define hyperparameter
    const int64_t input_size = 4;
    const int64_t output_size = 1;
    const int num_epochs = 10;
    const double learning_rate = 0.001;

    // Sample dataset
    auto x_train = torch::randint(0, 10, {15, 4});
    auto y_train = torch::randint(0, 10, {15, 1});

    // Linear regression model 
    torch::nn::Linear model(input_size, output_size);

    // Optimizer 
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(learning_rate));

    std::cout << "Training... \n";
    for(int epoch=0; epoch <= num_epochs; epoch++){
        // Forward pass 
        auto output = model(x_train);
        auto loss = torch::nn::functional::mse_loss(output, y_train);

        // Backward pass 
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        if ((epoch + 1) % 5 == 0) {
            std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs <<
                "], Loss: " << loss.item<double>() << "\n";
            }
        }
    std::cout << "Training finished!\n";
}