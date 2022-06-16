//
// Created by Esteban on 6/7/2022.
//

#include "MSELoss.h"

using namespace mlp::math;

MSELoss::Matrix MSELoss::forward (const MSELoss::Matrix& input_) {
#if defined(MLP_USE_BOOST_BACKEND)
    this->output = input_ * 1.0; // idk why this is needed but it yells at me otherwise
#elif defined(MLP_USE_ARRAYFIRE_BACKEND)
    this->output = input_;
#endif
    return this->output;
}

MSELoss::Matrix MSELoss::backward (const MSELoss::Matrix& real_value_, double) {
    return this->output - real_value_;
}

double MSELoss::loss (const MSELoss::Matrix& real_value_) const {
    const auto diff = this->output - real_value_;
    const auto mse = diff * diff;
#if defined(MLP_USE_BOOST_BACKEND)
    const auto ones = mlp::math::full(mse.size1(), 1.0);
    const auto sum_cols = ones % mse;
    return mlp::math::sum(sum_cols) / static_cast<double>(output.size2());
#elif defined(MLP_USE_ARRAYFIRE_BACKEND)
    mlp::math::eval(mse);
    return mlp::math::sum<double>(mse) / (double) mse.dims(1);
#endif
}
