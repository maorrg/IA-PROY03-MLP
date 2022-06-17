#include <mlp/net/DenseLayer.h>
#include <algorithm>
#include <random>
#include <boost/numeric/ublas/io.hpp>

using namespace mlp::math;

#if defined(MLP_USE_BOOST_BACKEND)
DenseLayer::DenseLayer (size_t input_size, size_t output_size) {
    this->weights = mlp::math::randu(output_size, input_size);
    this->biases = mlp::math::randu(output_size, 1);
    std::transform(this->weights.data().begin(), this->weights.data().end(), this->weights.data().begin(), [](auto x){ return x - 0.5; });
    std::transform(this->biases.data().begin(), this->biases.data().end(), this->biases.data().begin(), [](auto x){ return x - 0.5; });
}

DenseLayer::Matrix DenseLayer::forward (const DenseLayer::Matrix& input_) {
    this->input = input_;
    Matrix result = mlp::math::prod(this->weights, this->input);
    for (size_t i = 0; i < result.size2(); ++i) {
        mlp::math::column(result, i) += mlp::math::column(this->biases, 0);
    }
    return result;
}

DenseLayer::Matrix DenseLayer::backward (const DenseLayer::Matrix& gradient, double learning_rate) {
    const auto m = gradient.size2();

    Matrix next_gradient = prod(trans(this->weights), gradient);
    const Matrix weights_gradient = prod(gradient, trans(this->input)) / (double) m;
    const Matrix bias_gradient = prod(gradient, constant(m, 1, 1.0)) / (double) m;

    this->weights -= weights_gradient * learning_rate;
    this->biases -= bias_gradient * learning_rate;
    return next_gradient;
}

#elif defined(MLP_USE_ARRAYFIRE_BACKEND)
DenseLayer::DenseLayer (size_t input_size, size_t output_size) {
    this->weights = mlp::math::randu(output_size, input_size) - 0.5;
    this->biases = mlp::math::randu(output_size, 1) - 0.5;
}

DenseLayer::Matrix DenseLayer::forward (const DenseLayer::Matrix& input_) {
    this->input = input_;
    auto wx = mlp::math::matmul(this->weights, this->input);
    auto wx_plus_b = mlp::math::batchFunc(wx, this->biases, [](const Matrix& ax, const Matrix& b) { return ax + b; });
    mlp::math::eval(wx, wx_plus_b);
    return wx_plus_b;
}

DenseLayer::Matrix DenseLayer::backward (const DenseLayer::Matrix& gradient, double learning_rate) {
    const auto m = gradient.dims(1);

    Matrix next_gradient = mlp::math::matmulTN(this->weights, gradient);
    const Matrix weights_gradient = mlp::math::matmulNT(gradient, this->input) / (double) m;
    const Matrix bias_gradient = mlp::math::sum(gradient, 1) / (double) m;

    this->weights -= weights_gradient * learning_rate;
    this->biases -= bias_gradient * learning_rate;
    return next_gradient;
}
#endif



