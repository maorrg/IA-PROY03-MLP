#include <mlp/activation/ReLU.h>
#include <mlp/math/mlp_math.h>


#if defined(MLP_USE_BOOST_BACKEND)
MLP_MATH_MAKE_FUNCTOR(relu, [] (double x) { return double(x > 0) * x; });
ReLU::Matrix ReLU::relu (const ReLU::Matrix& input_) {
    return mlp::math::relu(input_);
}

MLP_MATH_MAKE_FUNCTOR(relu_derivative, [] (double x) { return double(x > 0); });
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

ReLU::Matrix ReLU::forward (const ReLU::Matrix& input_) {
    this->input = input_;
    auto output = ReLU::relu(input_);
    mlp::math::eval(output);
    return output;
}
ReLU::Matrix ReLU::backward (const ReLU::Matrix& gradient, double) {
    auto output = ReLU::relu_derivative(this->input) * gradient;
    mlp::math::eval(output);
    return output;
}
const char* ReLU::name () const { return "ReLU"; }
