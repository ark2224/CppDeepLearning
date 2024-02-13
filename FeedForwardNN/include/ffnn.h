#ifndef FFNN_H
#define FFNN_H
#include <torch/torch.h>

class FeedForwardNeuralNet : public torch::nn::Module {
public:
    FeedForwardNeuralNet(int64_t input_size, int64_t hidden_size, int64_t num_classes) : 
        fc1(input_size, hidden_size), fc2(hidden_size, num_classes) {
            register_module("fc1", fc1);
            register_module("fc2", fc2);
        }

    torch::Tensor forward(torch::Tensor x);

    static int build_ffnn();

private:
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
};

// TORCH_MODULE(FFNN)
#endif