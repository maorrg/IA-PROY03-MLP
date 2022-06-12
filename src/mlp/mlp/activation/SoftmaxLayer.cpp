//
// Created by Esteban on 6/4/2022.
//

#include <numeric>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/assign/std/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include "SoftmaxLayer.h"

namespace ub = boost::numeric::ublas;

template<class OP, class E>
BOOST_UBLAS_INLINE
typename ub::matrix_unary1_traits<E, OP>::result_type
apply_op (const ub::matrix_expression<E> &e) {
    typedef typename ub::matrix_unary1_traits<E, OP>::expression_type expression_type;
    return expression_type (e ());
}

SoftmaxLayer::Matrix SoftmaxLayer::forward (const SoftmaxLayer::Matrix& input_) {
    Matrix exps = input_;
    std::transform(exps.data().begin(), exps.data().end(), exps.data().begin(), [] (double x) { return std::exp(x); });
    for (size_t i = 0; i < exps.size2(); ++i) {
        ub::column(exps, i) /= ub::sum(ub::column(exps, i));
    }
    this->output = exps;
    return this->output;
}

SoftmaxLayer::Matrix SoftmaxLayer::backward (const SoftmaxLayer::Matrix& gradient, double) {
    // SUS
    const auto n = this->output.size1();
    const auto m = this->output.size2();
    Matrix result{n, m};
    for (size_t i = 0; i < m; ++i) {
        Matrix M{n, n};
        for (size_t j = 0; j < n; ++j) ub::column(M, j) = ub::column(this->output, i);
        const auto I = ub::identity_matrix<double>(n);
        ub::column(result, i) = ub::column(ub::prod(ub::element_prod(M, I - ub::trans(M)), gradient), 0);
    }
    return result;

}
