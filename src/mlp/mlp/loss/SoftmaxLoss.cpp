//
// Created by Esteban on 6/7/2022.
//

#include "SoftmaxLoss.h"

using namespace mlp::math;


MLP_BOOST_MATH_MAKE_FUNCTOR(exp, [] (double x) { return std::exp(x); });
SoftmaxLoss::Matrix SoftmaxLoss::forward (const Matrix& input_) {
    Matrix exps = mlp::math::exp(input_);
    for (size_t i = 0; i < exps.size2(); ++i) {
        mlp::math::col(exps, i) /= mlp::math::sum(mlp::math::col(exps, i));
    }
    this->output = exps;
    return this->output;
}

SoftmaxLoss::Matrix SoftmaxLoss::backward (const SoftmaxLoss::Matrix& real_value_, double) {
    // the fact that the derivative is the same as MSE if you assume Softmax is cool af
    return this->output - real_value_;
}

MLP_BOOST_MATH_MAKE_FUNCTOR(log_epsilon, [] (double x) { return std::log(x + 1e-8); });
double SoftmaxLoss::loss (const SoftmaxLoss::Matrix& real_value_) const {
    const auto logs = mlp::math::log_epsilon(this->output);
    const auto ce = - (real_value_ * logs);
    const auto ones = mlp::math::full(this->output.size1(), 1.0);
    const auto sum_cols = ones % ce;
    return mlp::math::sum(sum_cols) / static_cast<double>(this->output.size2());
}


