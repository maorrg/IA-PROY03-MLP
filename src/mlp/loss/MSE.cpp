//
// Created by Esteban on 6/5/2022.
//

#include "MSE.h"
#include <boost/numeric/ublas/matrix_proxy.hpp>

namespace ub = boost::numeric::ublas;

MSE::MSE () : LossFunction(MSE::mse_loss, MSE::mse_derivative) {}

double MSE::mse_loss (const LossFunction::Matrix& output, const LossFunction::Matrix& target) {
    double result = 0;
    const auto diff = output - target;
    for (size_t i = 0; i < diff.size2(); ++i) {
        result += ub::norm_2(ub::column(diff, i));
    }
    return result / static_cast<double>(diff.size2());
}

LossFunction::Matrix MSE::mse_derivative (const LossFunction::Matrix& output, const LossFunction::Matrix& target) {
    return output - target;
}
