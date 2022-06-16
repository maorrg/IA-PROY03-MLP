#include <mlp/activation/ReLU.h>
#include <mlp/math/mlp_math.h>

ReLU::ReLU () : ActivationLayer(ReLU::relu, ReLU::relu_derivative) {}

#if defined(MLP_USE_BOOST_BACKEND)
MLP_MATH_MAKE_FUNCTOR(relu, [] (double x) { return x > 0 ? x : 0; });
ReLU::Matrix ReLU::relu (const ReLU::Matrix& input_) {
    return mlp::math::relu(input_);
}

MLP_MATH_MAKE_FUNCTOR(relu_derivative, [] (double x) { return x > 0 ? 1 : 0; });
ReLU::Matrix ReLU::relu_derivative (const ReLU::Matrix& input_) {
    return mlp::math::relu_derivative(input_);
}

#elif defined(MLP_USE_ARRAYFIRE_BACKEND)
ReLU::Matrix ReLU::relu (const ReLU::Matrix& input_) {
    return (input_ > 0).as(f32) * input_;
}

ReLU::Matrix ReLU::relu_derivative (const ReLU::Matrix& input_) {
    return (input_ > 0).as(f32);
}

#endif
