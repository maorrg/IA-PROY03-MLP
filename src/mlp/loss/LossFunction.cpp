//
// Created by Esteban on 6/5/2022.
//

#include "LossFunction.h"
#include "../utils/Utilities.h"

namespace ub = boost::numeric::ublas;

double LossFunction::loss (const LossFunction::Matrix& output, const LossFunction::Matrix& target) const {
    return loss_function(output, target);
}
LossFunction::Matrix
LossFunction::derivative (const LossFunction::Matrix& output, const LossFunction::Matrix& target) const {
    return derivative_function(output, target);
}

LossFunction::LossFunction (LossFunction::LossFunc loss_function, LossFunction::LossGradient derivative_function) {
    this->loss_function = std::move(loss_function);
    this->derivative_function = std::move(derivative_function);
}


