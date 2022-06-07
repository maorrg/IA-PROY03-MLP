//
// Created by Esteban on 6/5/2022.
//

#include "../utils/Utilities.h"
#include "CrossEntropy.h"
#include <boost/numeric/ublas/matrix_proxy.hpp>

namespace ub = boost::numeric::ublas;

CrossEntropy::CrossEntropy () : LossFunction(
    CrossEntropy::cross_entropy_loss, CrossEntropy::cross_entropy_derivative) {}

double CrossEntropy::cross_entropy_loss (const LossFunction::Matrix& output, const LossFunction::Matrix& target) {
    double result = 0;
    const auto log_out = utils::foreach(output, [] (double x) { return std::log(1e-15 + x); });
    const auto comp_log_out = utils::foreach(output, [] (double x) { return std::log(1.0 - (1e-15 + x)); });
    const auto one = utils::ones(output.size1(), output.size2());

    auto result_matrix = -ub::element_prod(target, log_out) - ub::element_prod((one - target), comp_log_out);
    for (size_t i = 0; i < result_matrix.size2(); ++i) {
        result += ub::sum(ub::column(result_matrix, i));
    }
    return result / static_cast<double>(result_matrix.size2());
}

LossFunction::Matrix
CrossEntropy::cross_entropy_derivative (const LossFunction::Matrix& output, const LossFunction::Matrix& target) {
    Matrix result(output.size1(), output.size2());
    const auto one = utils::ones(output.size1(), output.size2());
    const auto small = utils::foreach(output, [] (double) { return 1e-15; });

    return -ub::element_div(target, output + small) + ub::element_div((one - target), (one - (output + small)));
}
