//
// Created by Esteban on 6/16/2022.
//

#include "NetworkReLU.h"

NetworkReLU::NetworkReLU (size_t input_size, size_t output_size, const NetworkSettings& settings)
    : Network(settings),
      dense1(input_size, 10),
      dense2(10, 10),
      dense3(10, output_size) {
}

NetworkReLU::Matrix NetworkReLU::forward (const NetworkReLU::Matrix& input) {
    auto x = dense1.forward(input);
    x = relu1.forward(x);
    x = dense2.forward(x);
    x = relu2.forward(x);
    x = dense3.forward(x);
    x = softmax.forward(x);
    return x;
}

void NetworkReLU::backward (const NetworkReLU::Matrix& real_value) {
    loss = softmax.loss(real_value);
    auto x = softmax.backward(real_value, settings.learning_rate);
    x = dense3.backward(x, settings.learning_rate);
    x = relu2.backward(x, settings.learning_rate);
    x = dense2.backward(x, settings.learning_rate);
    x = relu1.backward(x, settings.learning_rate);
    dense1.backward(x, settings.learning_rate);
}

void NetworkReLU::on_epoch_callback (size_t epoch) {
    std::cout << "Epoch: " << epoch << ' ' << "Loss: " << loss << std::endl;
}
