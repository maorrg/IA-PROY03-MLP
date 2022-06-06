//
// Created by Esteban on 6/4/2022.
//

#ifndef UNTITLED21_NETWORK_H
#define UNTITLED21_NETWORK_H

#include <utility>

#include "layer/Layer.h"
#include "loss/LossFunction.h"

class Network {
public:
    using Matrix = boost::numeric::ublas::matrix<double>;

    explicit Network (std::vector<Layer*> layers_, LossFunction lossFunction_)
        : layers(std::move(layers_)),
          lossFunction(std::move(lossFunction_)) {
    }

    Network& train(const Matrix& input, const Matrix& target, size_t epochs, double learning_rate) {
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            Matrix output = this->predict(input);
            std::cout << "Epoch: " << epoch << ", Loss: " << lossFunction.loss(output, target) << std::endl;
            auto gradient = lossFunction.derivative(output, target);
            for (size_t i = layers.size(); i > 0; --i) gradient = layers[i - 1]->backward(gradient, learning_rate);
        }
        return *this;
    }

    Matrix predict(const Matrix& input) {
        Matrix output = input;
        for (size_t i = 0; i < layers.size(); ++i) output = layers[i]->forward(output);
        return output;
    }
private:
    // non owning
    std::vector<Layer*> layers;
    LossFunction lossFunction;
};


#endif //UNTITLED21_NETWORK_H
