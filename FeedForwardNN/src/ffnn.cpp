// #include <ffnn.h>
#include "../include/ffnn.h"
#include <torch/torch.h>


torch::Tensor FeedForwardNeuralNet::forward(torch::Tensor x) {
    x = torch::nn::functional::relu(fc1->forward(x));
    return fc2->forward(x);
}


int FeedForwardNeuralNet::build_ffnn() {
    std::cout << "Creating Feed-forward Neural Net" << std::endl;
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << std::endl;

    const int64_t input_size = 1024;
    const int64_t hidden_dim = 512;
    const int64_t num_classes = 10;
    const int64_t batch_size = 64;
    const size_t total_epochs = 3;
    const double learning_rate = 0.001;

    std::cout << "here1" << std::endl;
    auto train_dataset = torch::data::datasets::MNIST("./data")
        .map(torch::data::transforms::Normalize<>(0.5, 0.5))
        .map(torch::data::transforms::Stack<>());
    std::cout << "here2" << std::endl;
    auto test_dataset = torch::data::datasets::MNIST("./data")
        .map(torch::data::transforms::Normalize<>(0.5, 0.5))
        .map(torch::data::transforms::Stack<>());
    
    auto num_training_samples = train_dataset.size().value();
    auto num_test_samples = test_dataset.size().value();

    auto train_loader = torch::data::make_data_loader(
        std::move(train_dataset),
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(2));
    for (torch::data::Example<>& batch : *train_loader) {
        std::cout << "Batch size: " << batch.data.size(0) << " | Labels: ";
        for (int64_t i = 0; i < batch.data.size(0); ++i) {
            std::cout << batch.target[i].item<int64_t>() << " ";
        }
        std::cout << std::endl;
    }

    auto test_loader = torch::data::make_data_loader(
        std::move(test_dataset),
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(2));

    // auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
    //     std::move(train_dataset), batch_size
    // );
    // auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
    //     std::move(test_dataset), batch_size
    // );

    FeedForwardNeuralNet model = FeedForwardNeuralNet(input_size, hidden_dim, num_classes);
    model.to(device);

    torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(learning_rate));

    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Commencing Training ...\n";

    for (size_t epoch = 0; epoch != total_epochs; ++epoch) {
        double running_loss = 0.0;
        size_t num_correct = 0;
        for (auto& batch : *train_loader) {
            auto data = batch.data.view({batch_size, -1}).to(device);
            auto target = batch.target.to(device);

            auto output = model.forward(data);
            auto loss = torch::nn::functional::cross_entropy(output, target);

            running_loss += loss.item<double>() * data.size(0);

            auto prediction = output.argmax(1);

            num_correct += prediction.eq(target).sum().item<int64_t>();

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }

        auto sample_mean_loss = running_loss / num_training_samples;
        auto accuracy = static_cast<double>(num_correct) / num_training_samples;

        std::cout << "Epoch [" << (epoch + 1) << "/" << total_epochs << "], Trainset - Loss: "
            << sample_mean_loss << ", Accuracy: " << accuracy << "\n";
    }

    std::cout << "Training finished\n";
    std::cout << "Commencing Testing ...\n";

    model.eval();
    torch::NoGradGuard no_grad;

    double running_loss = 0.0;
    size_t num_correct = 0;

    for (auto& batch : *test_loader) {
        auto data = batch.data.view({batch_size, -1}).to(device);
        auto target = batch.target.to(device);

        auto output = model.forward(data);
        auto loss = torch::nn::functional::cross_entropy(output, target);

        running_loss += loss.item<double>() * data.size(0);

        auto prediction = output.argmax(1);

        num_correct += prediction.eq(target).sum().item<int64_t>();
    }

    std::cout << "Testing finished.\n";
    auto test_sample_mean_loss = running_loss / num_test_samples;
    auto test_accuracy = static_cast<double>(num_correct) / num_test_samples;

    std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << "\n";
    return 0;
}