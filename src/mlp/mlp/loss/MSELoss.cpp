//
// Created by Esteban on 6/7/2022.
//

#include "MSELoss.h"

namespace ub = boost::numeric::ublas;

MSELoss::Matrix MSELoss::forward (const MSELoss::Matrix& input_) {
    this->output = input_ * 1.0;
    return this->output;
}

MSELoss::Matrix MSELoss::backward (const MSELoss::Matrix& real_value_, double) {
    return this->output - real_value_;
}

double MSELoss::loss (const MSELoss::Matrix& real_value_) const {
    const auto diff = this->output - real_value_;
    const auto mse = ub::element_prod(diff, diff);
    const auto sum_cols = ub::prod(ub::scalar_vector(output.size1(), 1.0), mse);
    return ub::sum(sum_cols) / static_cast<double>(output.size2());
}
