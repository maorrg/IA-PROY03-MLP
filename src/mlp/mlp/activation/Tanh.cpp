#include <mlp/activation/Tanh.h>

Tanh::Tanh () : ActivationLayer(Tanh::tanh, Tanh::tanh_derivative) {}

MLP_BOOST_MATH_MAKE_FUNCTOR(tanh, [] (double x) { return std::tanh(x); });
Tanh::Matrix Tanh::tanh (const Tanh::Matrix& input_) {
    return mlp::math::tanh(input_);
}

MLP_BOOST_MATH_MAKE_FUNCTOR(tanh_derivative, [] (double x) { return 1 - std::pow(std::tanh(x), 2); });
Tanh::Matrix Tanh::tanh_derivative (const Tanh::Matrix& input_) {
    return mlp::math::tanh_derivative(input_);
}
