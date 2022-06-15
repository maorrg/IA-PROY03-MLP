//
// Created by Esteban on 6/5/2022.
//

#include <mlp/activation/SigmoidLayer.h>

SigmoidLayer::SigmoidLayer () : ActivationLayer (SigmoidLayer::sigmoid, SigmoidLayer::sigmoid_derivative) {}

MLP_BOOST_MATH_MAKE_FUNCTOR(sigmoid, [] (double x) { return 1 / (1 + std::exp(-x)); });
SigmoidLayer::Matrix SigmoidLayer::sigmoid (const SigmoidLayer::Matrix& input_) {
    return mlp::math::sigmoid(input_);
}

MLP_BOOST_MATH_MAKE_FUNCTOR(sigmoid_derivative, [] (double x) { return std::exp(-x) / std::pow(1 + std::exp(-x), 2); });
SigmoidLayer::Matrix SigmoidLayer::sigmoid_derivative (const SigmoidLayer::Matrix& input_) {
    return mlp::math::sigmoid_derivative(input_);
}

