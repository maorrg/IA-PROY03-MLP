//
// Created by Esteban on 6/5/2022.
//

#include "ReLU.h"

ReLU::ReLU () : ActivationLayer (ReLU::relu, ReLU::relu_derivative) {}

ReLU::Matrix ReLU::relu (const ReLU::Matrix& input_) {
    Matrix output = input_;
    std::transform(output.data().begin(), output.data().end(), output.data().begin(), [](double x) -> double {
        return x > 0.0 ? x : 1e-3 * x;
    });
    return output;
}
ReLU::Matrix ReLU::relu_derivative (const ReLU::Matrix& input_) {
    Matrix output = input_;
    std::transform(output.data().begin(), output.data().end(), output.data().begin(), [](double x) -> double {
        return x > 0.0 ? 1.0 : 1e-3;
    });
    return output;
}

