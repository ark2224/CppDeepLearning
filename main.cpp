#include <torch/torch.h>


struct CVAE1 : torch::nn::Module {
    CVAE1() {
        cvn1 = register_module("cvn1", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(1, 64, 1).stride(2).padding('same').bias(true)));
        // left off right here trying to figure out the parameters to the above conv2d layer^
        
        
        
        
        cvn2 = register_module("cvn2", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(1, 680, 64).stride(2).padding('same').bias(true)));
        cvn3 = register_module("cvn3", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(1, 340, 32).stride(2).padding('same').bias(true)));

        fc1 = register_module("fc1", torch::nn::Linear(170));
        fc2 = register_module("fc2", torch::nn::Linear(64, 32));
        
        
        Encoder(
            cvn1,//input is (1, 1360, 1)
            torch::nn::ReLU(),
            torch::nn::BatchNorm2d(),//what to put as parameters?
            cvn2,
            torch::nn::ReLU(),
            torch::nn::BatchNorm2d(),//what to put as parameters?
            cvn3,
            torch::nn::ReLU(),
            torch::nn::BatchNorm2d(),//what to put as parameters?
            // torch::nn::reshape()
            torch::nn::Flatten(),
            fc1,
            fc2
        );
        cvn1 = register_module("cvn1", torch::nn::Conv2d());



    }

    torch::Tensor forward(torch::Tensor x) {
        x = Encoder->forward(x);
        x = x.reshape({1, 1, 170});
        x = torch::relu(fc1->forward(x.reshape({1, 170})));
        x = torch::dropout(x, 0.5, is_training());
        return x;
    }

    torch::nn::Sequential Encoder{nullptr}, Decoder{nullptr};

    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    torch::nn::Conv2d cvn1{nullptr}, cvn2{nullptr}, cvn3{nullptr};
};

// int main() {
//     torch::Tensor tensor = torch::eye(3);
//     std::cout << tensor << std::endl;
// }

int main() {
  // Create a new cvae1.
  auto cvae1 = std::make_shared<CVAE1>();

  // Create a multi-threaded data loader for the MNIST dataset.
  auto data_loader = torch::data::make_data_loader(
      torch::data::datasets::MNIST("./data").map(
          torch::data::transforms::Stack<>()),
      /*batch_size=*/64);

  // Instantiate an SGD optimization algorithm to update our cvae1's parameters.
  torch::optim::SGD optimizer(cvae1->parameters(), /*lr=*/0.01);

  for (size_t epoch = 1; epoch <= 10; ++epoch) {
    size_t batch_index = 0;
    // Iterate the data loader to yield batches from the dataset.
    for (auto& batch : *data_loader) {
      // Reset gradients.
      optimizer.zero_grad();
      // Execute the model on the input data.
      torch::Tensor prediction = cvae1->forward(batch.data);
      // Compute a loss value to judge the prediction of our model.
      torch::Tensor loss = torch::nll_loss(prediction, batch.target);
      // Compute gradients of the loss w.r.t. the parameters of our model.
      loss.backward();
      // Update the parameters based on the calculated gradients.
      optimizer.step();
      // Output the loss and checkpoint every 100 batches.
      if (++batch_index % 100 == 0) {
        std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                  << " | Loss: " << loss.item<float>() << std::endl;
        // Serialize your model periodically as a checkpoint.
        torch::save(cvae1, "cvae1.pt");
      }
    }
  }
}
