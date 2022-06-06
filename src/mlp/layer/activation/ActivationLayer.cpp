//
// Created by Esteban on 6/4/2022.
//

#include "ActivationLayer.h"

ActivationLayer::ActivationLayer (ActivationLayer::Function function_, ActivationLayer::Function derivative_) :
    function(std::move(function_)), derivative(std::move(derivative_)) {}

ActivationLayer::Matrix ActivationLayer::forward (const ActivationLayer::Matrix& input_) {
    this->input = input_;
    return this->function(input_);
}
ActivationLayer::Matrix ActivationLayer::backward (const ActivationLayer::Matrix& gradient, double) {
    return boost::numeric::ublas::element_prod(this->derivative(this->input), gradient);
}
