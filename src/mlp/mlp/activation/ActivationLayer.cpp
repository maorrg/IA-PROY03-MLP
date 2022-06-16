//
// Created by Esteban on 6/4/2022.
//

#include <mlp/activation/ActivationLayer.h>

using namespace mlp::math;

ActivationLayer::ActivationLayer (ActivationLayer::Function function_, ActivationLayer::Function derivative_) :
    function(function_), derivative(derivative_) {}

ActivationLayer::Matrix ActivationLayer::forward (const ActivationLayer::Matrix& input_) {
    this->input = input_;
    auto output = this->function(input_);
#if defined(MLP_USE_ARRAYFIRE_BACKEND)
    mlp::math::eval(output);
#endif
    return output;
}
ActivationLayer::Matrix ActivationLayer::backward (const ActivationLayer::Matrix& gradient, double) {
    return this->derivative(this->input) * gradient;
}
