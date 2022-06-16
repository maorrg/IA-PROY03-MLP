#include <mlp/activation/Tanh.h>


#if defined(MLP_USE_BOOST_BACKEND)
MLP_MATH_MAKE_FUNCTOR(tanh, [] (double x) { return std::tanh(x); });
Tanh::Matrix Tanh::tanh (const Tanh::Matrix& input_) {
    return mlp::math::tanh(input_);
}

MLP_MATH_MAKE_FUNCTOR(tanh_derivative, [] (double x) { return 1 - std::pow(x, 2); });
Tanh::Matrix Tanh::tanh_derivative (const Tanh::Matrix& output_) {
    return mlp::math::tanh_derivative(output_);
}
#elif defined(MLP_USE_ARRAYFIRE_BACKEND)
Tanh::Matrix Tanh::tanh (const Tanh::Matrix& input_) {
    return (mlp::math::tanh(input_)).as(f32);
}

Tanh::Matrix Tanh::tanh_derivative (const Tanh::Matrix& output_) {
    return (1.0 - mlp::math::pow(output_, 2.0)).as(f32);
}
#endif

Tanh::Matrix Tanh::forward (const Tanh::Matrix& input_) {
    this->output = Tanh::tanh(input_);
    mlp::math::eval(this->output);
    return this->output;
}

Tanh::Matrix Tanh::backward (const Tanh::Matrix& gradient, double) {
    auto n_gradient = Tanh::tanh_derivative(this->output) * gradient;
    mlp::math::eval(n_gradient);
    return n_gradient;
}