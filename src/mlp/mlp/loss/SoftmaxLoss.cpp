//
// Created by Esteban on 6/7/2022.
//

#include "SoftmaxLoss.h"

using namespace mlp::math;

#if defined(MLP_USE_BOOST_BACKEND)
MLP_MATH_MAKE_FUNCTOR(exp, [] (double x) { return std::exp(x); });
SoftmaxLoss::Matrix SoftmaxLoss::forward (const Matrix& input_) {
    this->output = input_ * 1.0;
    for (size_t i = 0; i < input_.size2(); ++i) {
        auto column = mlp::math::column(this->output, i);
        auto max = std::reduce(column.begin(), column.end(), std::numeric_limits<double>::min(), [](auto a, auto b){ return std::max(a, b); });
        std::transform(column.begin(), column.end(), column.begin(), [max](auto x) { return exp(x - max); });
        column /= mlp::math::sum(column);
    }
    return this->output;
}

SoftmaxLoss::Matrix SoftmaxLoss::backward (const SoftmaxLoss::Matrix& real_value_, double) {
    return  this->output - real_value_;
}

MLP_MATH_MAKE_FUNCTOR(log, [] (double x) {return std::log(x + 1e-8);});
double SoftmaxLoss::loss (const SoftmaxLoss::Matrix& real_value_) const {
    const Matrix logs = mlp::math::log(this->output);
    const Matrix ce = -(real_value_ * logs);
    const auto sum = std::accumulate(ce.data().begin(), ce.data().end(), 0.0, [](auto a, auto b){ return a + b; });
    return sum / static_cast<double>(this->output.size2());
}

#elif defined(MLP_USE_ARRAYFIRE_BACKEND)
#undef max
SoftmaxLoss::Matrix SoftmaxLoss::forward (const Matrix& input_) {
    const auto subtract_max = [] (const Matrix& x, const Matrix& m) { return x - m; };
    const auto ex_div_ex_sum = [] (const Matrix& ex_sum, const Matrix& ex) { return ex / ex_sum; };
    Matrix max_per_col = mlp::math::max(input_, 0);
    Matrix norm = mlp::math::batchFunc(input_, max_per_col, subtract_max);
    Matrix exps = mlp::math::exp(norm);
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
    const auto ce = -(real_value_ * mlp::math::log(this->output + 1e-12));
    auto result = mlp::math::sum<double>(ce) / (double) ce.elements();
    return result;
}
#endif


