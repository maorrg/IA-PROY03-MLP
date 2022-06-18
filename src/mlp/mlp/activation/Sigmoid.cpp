//
// Created by Esteban on 6/5/2022.
//

#include <mlp/activation/Sigmoid.h>


#if defined(MLP_USE_BOOST_BACKEND)
MLP_MATH_MAKE_FUNCTOR(sigmoid, [] (double x) { return 1 / (1 + std::exp(-x)); });
Sigmoid::Matrix Sigmoid::sigmoid (const Sigmoid::Matrix& input_) {
    return mlp::math::sigmoid(input_);
}

MLP_MATH_MAKE_FUNCTOR(sigmoid_derivative, [] (double x) { return x * (1.0 - x); });
Sigmoid::Matrix Sigmoid::sigmoid_derivative (const Sigmoid::Matrix& output_) {
    return mlp::math::sigmoid_derivative(output_);
}
#elif defined(MLP_USE_ARRAYFIRE_BACKEND)
Sigmoid::Matrix Sigmoid::sigmoid (const Sigmoid::Matrix& input_) {
    auto output = (1.0 / (1.0 + mlp::math::exp(-input_))).as(f32);
    mlp::math::eval(output);
    return output;
}

Sigmoid::Matrix Sigmoid::sigmoid_derivative (const Sigmoid::Matrix& output_) {
    return output_ * (1.0 - output_);
}
#endif

Sigmoid::Matrix Sigmoid::forward (const Sigmoid::Matrix& input_) {
    this->output = Sigmoid::sigmoid(input_);
    mlp::math::eval(this->output);
    return this->output;
}

Sigmoid::Matrix Sigmoid::backward (const Sigmoid::Matrix& gradient, double) {
    auto n_gradient = Sigmoid::sigmoid_derivative(this->output) * gradient;
    mlp::math::eval(n_gradient);
    return n_gradient;
}
const char* Sigmoid::name () const { return "Sigmoid"; }



