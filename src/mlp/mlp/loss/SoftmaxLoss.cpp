//
// Created by Esteban on 6/7/2022.
//

#include "SoftmaxLoss.h"
#include <boost/numeric/ublas/matrix_proxy.hpp>

namespace ub = boost::numeric::ublas;


SoftmaxLoss::SoftmaxLoss (double epsilon)
    : LossLayer(), epsilon(epsilon) {}

SoftmaxLoss::Matrix SoftmaxLoss::forward (const Matrix& input_) {
    Matrix exps = input_;
    std::transform(exps.data().begin(), exps.data().end(), exps.data().begin(), [] (double x) { return std::exp(x); });
    for (size_t i = 0; i < exps.size2(); ++i) {
        ub::column(exps, i) /= ub::sum(ub::column(exps, i));
    }
    this->output = exps;
    return this->output;
}

SoftmaxLoss::Matrix SoftmaxLoss::backward (const SoftmaxLoss::Matrix& real_value_, double) {
    // the fact that the derivative is the same as MSE if you assume Softmax is cool af
    return this->output - real_value_;
}

double SoftmaxLoss::loss (const SoftmaxLoss::Matrix& real_value_) const {
    Matrix log_out = this->output;
    std::transform(
        log_out.data().begin(), log_out.data().end(), log_out.data().begin(),
        [this] (double x) { return std::log(x + epsilon); });
    const auto ce = -ub::element_prod(real_value_, log_out);
    const ub::vector<double> sum_cols = ub::prod(ub::scalar_vector(this->output.size1(), 1.0), ce);
    return ub::sum(sum_cols) / static_cast<double>(this->output.size2());
}

