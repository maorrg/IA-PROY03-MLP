//
// Created by Esteban on 6/4/2022.
//

#include "Network.h"


Network& Network::train (const Matrix& input, const Matrix& target, size_t epochs, double learning_rate) {
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        this->forward(input);
        std::cout << "Epoch:\t" << epoch << ", Loss: " << loss_layer->loss(target) << std::endl;
        backward(target, learning_rate);
    }
    return *this;
}

Network::Matrix Network::forward (const Matrix& input) {
    Matrix output = input;
    for (auto & hidden_layer : hidden_layers) {
        output = hidden_layer->forward(output);
    }
    output = loss_layer->forward(output);
    return output;
}

void Network::backward (const Network::Matrix& real_value, double learning_rate) {
    auto gradient = loss_layer->backward(real_value, learning_rate);
    for (size_t i = hidden_layers.size(); i > 0; --i) {
        gradient = hidden_layers[i - 1]->backward(gradient, learning_rate);
    }
}

Network::Network (std::vector<Layer*> hidden_layers_, LossLayer* loss_layer) {
    this->hidden_layers = std::move(hidden_layers_);
    this->loss_layer = loss_layer;
}



