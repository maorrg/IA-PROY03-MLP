//
// Created by Esteban on 6/5/2022.
//

#include <mlp/activation/Sigmoid.h>

Sigmoid::Sigmoid () : ActivationLayer (Sigmoid::sigmoid, Sigmoid::sigmoid_derivative) {}

#if defined(MLP_USE_BOOST_BACKEND)
MLP_MATH_MAKE_FUNCTOR(sigmoid, [] (double x) { return 1 / (1 + std::exp(-x)); });
Sigmoid::Matrix Sigmoid::sigmoid (const Sigmoid::Matrix& input_) {
    return mlp::math::sigmoid(input_);
}

MLP_MATH_MAKE_FUNCTOR(sigmoid_derivative, [] (double x) { return std::exp(-x) / std::pow(1 + std::exp(-x), 2); });
Sigmoid::Matrix Sigmoid::sigmoid_derivative (const Sigmoid::Matrix& input_) {
    return mlp::math::sigmoid_derivative(input_);
}
#elif defined(MLP_USE_ARRAYFIRE_BACKEND)
Sigmoid::Matrix Sigmoid::sigmoid (const Sigmoid::Matrix& input_) {
    return (1.0 / (1.0 + mlp::math::exp(-input_))).as(f32);
}

Sigmoid::Matrix Sigmoid::sigmoid_derivative (const Sigmoid::Matrix& input_) {
    return mlp::math::exp(-input_) / mlp::math::pow(1.0 + mlp::math::exp(-input_), 2.0).as(f32);
}
#endif

