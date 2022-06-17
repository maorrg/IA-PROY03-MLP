//
// Created by Esteban on 6/16/2022.
//

#include "NetworkReLU.h"

NetworkReLU::NetworkReLU (size_t input_size, size_t output_size, const NetworkSettings& settings)
    : Network(settings),
      dense1(input_size, 32),
      dense2(32, 32),
      dense3(32, output_size) {
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
    this->loss = this->calculate_loss(real_value);
    auto x = softmax.backward(real_value, settings.learning_rate);
    x = dense3.backward(x, settings.learning_rate);
    x = relu2.backward(x, settings.learning_rate);
    x = dense2.backward(x, settings.learning_rate);
    x = relu1.backward(x, settings.learning_rate);
    dense1.backward(x, settings.learning_rate);
}

void NetworkReLU::on_epoch_callback (size_t epoch) {
    on_epoch_callback_(this, epoch);
}

double NetworkReLU::calculate_loss (const NetworkReLU::Matrix& real_value) const {
    return softmax.loss(real_value);
}
