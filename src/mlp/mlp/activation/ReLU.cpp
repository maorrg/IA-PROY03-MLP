#include <mlp/activation/ReLU.h>
#include <mlp/math/boost_math.h>

ReLU::ReLU () : ActivationLayer(ReLU::relu, ReLU::relu_derivative) {}

MLP_BOOST_MATH_MAKE_FUNCTOR(relu, [] (double x) { return x > 0 ? x : 0; });
ReLU::Matrix ReLU::relu (const ReLU::Matrix& input_) {
    return mlp::math::relu(input_);
}

MLP_BOOST_MATH_MAKE_FUNCTOR(relu_derivative, [] (double x) { return x > 0 ? 1 : 0; });
ReLU::Matrix ReLU::relu_derivative (const ReLU::Matrix& input_) {
    return mlp::math::relu_derivative(input_);
}

