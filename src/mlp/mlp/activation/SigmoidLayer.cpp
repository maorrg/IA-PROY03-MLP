//
// Created by Esteban on 6/5/2022.
//

#include "SigmoidLayer.h"

SigmoidLayer::SigmoidLayer () : ActivationLayer (SigmoidLayer::sigmoid, SigmoidLayer::sigmoid_derivative) {}

SigmoidLayer::Matrix SigmoidLayer::sigmoid (const SigmoidLayer::Matrix& input_) {
    Matrix output = input_;
    std::transform(output.data().begin(), output.data().end(), output.data().begin(), [](double x) {
        return 1 / (1 + std::exp(-x));
    });
    return output;
}
SigmoidLayer::Matrix SigmoidLayer::sigmoid_derivative (const SigmoidLayer::Matrix& input_) {
    Matrix output = input_;
    std::transform(output.data().begin(), output.data().end(), output.data().begin(), [](double x) {
        return std::exp(-x) / std::pow(1 + std::exp(-x), 2);
    });
    return output;
}

