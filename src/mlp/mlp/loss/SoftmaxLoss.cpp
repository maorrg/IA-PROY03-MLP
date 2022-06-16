//
// Created by Esteban on 6/7/2022.
//

#include "SoftmaxLoss.h"

using namespace mlp::math;

#if defined(MLP_USE_BOOST_BACKEND)
MLP_MATH_MAKE_FUNCTOR(exp, [] (double x) { return std::exp(x); });
SoftmaxLoss::Matrix SoftmaxLoss::forward (const Matrix& input_) {
    Matrix exps = mlp::math::exp(input_);
    for (size_t i = 0; i < exps.size2(); ++i) {
        mlp::math::column(exps, i) /= mlp::math::sum(mlp::math::column(exps, i));
    }
    this->output = exps;
    return this->output;
}

SoftmaxLoss::Matrix SoftmaxLoss::backward (const SoftmaxLoss::Matrix& real_value_, double) {
    return  this->output - real_value_;
}

MLP_MATH_MAKE_FUNCTOR(log, [] (double x) {return std::log(x);});
double SoftmaxLoss::loss (const SoftmaxLoss::Matrix& real_value_) const {
    const auto logs = mlp::math::log(this->output);
    const auto ce = -(real_value_ * logs);
    const auto ones = mlp::math::full(this->output.size1(), 1.0);
    const auto sum_cols = ones % ce;
    return mlp::math::sum(sum_cols) / static_cast<double>(this->output.size2());
}

#elif defined(MLP_USE_ARRAYFIRE_BACKEND)
SoftmaxLoss::Matrix SoftmaxLoss::forward (const Matrix& input_) {
    const auto ex_div_ex_sum = [] (const Matrix& ex_sum, const Matrix& ex) { return ex / ex_sum; };

    Matrix exps = mlp::math::exp(input_);
    Matrix exp_sums = mlp::math::sum(exps, 0);
    this->output = mlp::math::batchFunc(exp_sums, exps, ex_div_ex_sum);;
    return this->output;
}

SoftmaxLoss::Matrix SoftmaxLoss::backward (const SoftmaxLoss::Matrix& real_value_, double) {
    auto diff = this->output - real_value_;
    mlp::math::eval(diff);
    return diff;
}

double SoftmaxLoss::loss (const SoftmaxLoss::Matrix& real_value_) const {
    const auto ce = -(real_value_ * mlp::math::log(this->output));
    mlp::math::eval(ce);
    auto result = mlp::math::sum<double>(ce) / (double) ce.dims(1);
    return result;
}
#endif


