#include "DenseLayer.h"
#include "../utils/Utilities.h"
#include <boost/numeric/ublas/tensor.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <algorithm>
#include <random>

namespace ub = boost::numeric::ublas;

DenseLayer::DenseLayer (size_t input_size, size_t output_size) {
    static auto random_device = std::random_device();
    static auto random_engine = std::default_random_engine{random_device()};
    static auto dist = std::uniform_real_distribution<double>(-1.0, 1.0);
    const auto generate = [&]() { return dist(random_engine); };

    this->weights = Matrix(output_size, input_size);
    this->biases = Matrix(output_size, 1);

    std::generate(this->weights.data().begin(), this->weights.data().end(), generate);
    std::generate(this->biases.data().begin(), this->biases.data().end(), generate);
}

DenseLayer::Matrix DenseLayer::forward (const DenseLayer::Matrix& input_) {
    this->input = input_;
    Matrix result = ub::prod(this->weights, input);
    for (size_t i = 0; i < result.size1(); ++i) {
        ub::column(result, i) += ub::column(this->biases, 0);
    }
    return result;
}

DenseLayer::Matrix DenseLayer::backward (const DenseLayer::Matrix& gradient, double learning_rate) {
    const auto m = gradient.size2();

    Matrix next_gradient = ub::prod(trans(this->weights), gradient);
    Matrix weights_gradient = ub::prod(gradient, trans(this->input)) / (double) m;
//    Matrix bias_gradient = sum_of_cols(gradient) / m;
    Matrix bias_gradient = ub::prod(gradient, utils::ones(m, 1)) / (double) m;

    this->weights -= weights_gradient * learning_rate;
    this->biases -= bias_gradient * learning_rate;
    return next_gradient;
}


