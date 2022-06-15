#include <mlp/net/DenseLayer.h>
#include <algorithm>
#include <random>

using namespace mlp::math;

inline double default_random () {
    static auto random_device = std::random_device();
    static auto random_engine = std::default_random_engine{ random_device() };
    static auto dist = std::uniform_real_distribution<double>(-0.5, 0.5);
    return dist(random_engine);
}

DenseLayer::DenseLayer (size_t input_size, size_t output_size)
    : DenseLayer(input_size, output_size, default_random) {}

DenseLayer::DenseLayer (size_t input_size, size_t output_size, const DenseLayer::Random& random) {
    this->weights = Matrix(output_size, input_size);
    this->biases = Matrix(output_size, 1);

    std::generate(this->weights.data().begin(), this->weights.data().end(), random);
    std::generate(this->biases.data().begin(), this->biases.data().end(), random);
}

DenseLayer::Matrix DenseLayer::forward (const DenseLayer::Matrix& input_) {
    this->input = input_;
    Matrix result = this->weights % this->input;
    for (size_t i = 0; i < result.size2(); ++i) {
        mlp::math::col(result, i) += mlp::math::col(this->biases, 0);
    }
    return result;
}

DenseLayer::Matrix DenseLayer::backward (const DenseLayer::Matrix& gradient, double learning_rate) {
    const auto m = gradient.size2();

    const Matrix next_gradient = mlp::math::trans(this->weights) % gradient;
    const Matrix weights_gradient = gradient % mlp::math::trans(this->input) / (double) m;
    const Matrix bias_gradient = gradient % mlp::math::full(m, 1, 1.0) / (double) m;

    this->weights -= weights_gradient * learning_rate;
    this->biases -= bias_gradient * learning_rate;
    return next_gradient;
}


