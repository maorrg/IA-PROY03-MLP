//
// Created by Esteban on 6/5/2022.
//

#include "ReluLayer.h"

ReluLayer::ReluLayer () : ActivationLayer (ReluLayer::relu, ReluLayer::relu_derivative) {}

ReluLayer::Matrix ReluLayer::relu (const ReluLayer::Matrix& input_) {
    Matrix output = input_;
    std::transform(output.data().begin(), output.data().end(), output.data().begin(), [](double x) -> double {
        return x > 0.0 ? x : 1e-3 * x;
    });
    return output;
}
ReluLayer::Matrix ReluLayer::relu_derivative (const ReluLayer::Matrix& input_) {
    Matrix output = input_;
    std::transform(output.data().begin(), output.data().end(), output.data().begin(), [](double x) -> double {
        return x > 0.0 ? 1.0 : 1e-3;
    });
    return output;
}

