//
// Created by Esteban on 6/4/2022.
//

#include "Network.h"


Network::Network(std::vector<Layer*> layers_, LossFunction lossFunction_)
    : layers(std::move(layers_)),
      lossFunction(std::move(lossFunction_)) {
}

Network& Network::train(const Matrix& input, const Matrix& target, size_t epochs, double learning_rate) {
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        Matrix output = this->predict(input);
        std::cout << "Epoch:\t" << epoch << ", Loss: " << lossFunction.loss(output, target) << std::endl;
        std::cout << "Output:\t" << output << std::endl;
        auto gradient = lossFunction.derivative(output, target);
        for (size_t i = layers.size(); i > 0; --i) {
            std::cout << gradient << '\t' << std::endl;
            gradient = layers[i - 1]->backward(gradient, learning_rate);
        }
        std::cout << std::endl;
    }
    return *this;
}

Network::Matrix Network::predict(const Matrix& input) {
    Matrix output = input;
    for (size_t i = 0; i < layers.size(); ++i) {
        output = layers[i]->forward(output);
    }
    return output;
}