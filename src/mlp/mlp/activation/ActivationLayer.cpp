//
// Created by Esteban on 6/4/2022.
//

#include <mlp/activation/ActivationLayer.h>

using namespace mlp::math;

ActivationLayer::ActivationLayer (ActivationLayer::Function function_, ActivationLayer::Function derivative_) :
    function(std::move(function_)), derivative(std::move(derivative_)) {}

ActivationLayer::Matrix ActivationLayer::forward (const ActivationLayer::Matrix& input_) {
    this->input = input_;
    return this->function(input_);
}
ActivationLayer::Matrix ActivationLayer::backward (const ActivationLayer::Matrix& gradient, double) {
    return this->derivative(this->input) * gradient;
}
