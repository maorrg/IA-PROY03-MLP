//
// Created by Esteban on 6/5/2022.
//

#include "Utilities.h"

utils::Matrix utils::ones (size_t size_1, size_t size_2) {
    boost::numeric::ublas::matrix<double> result(size_1, size_2);
    std::fill(result.data().begin(), result.data().end(), 1.0);
    return result;
}

utils::Matrix utils::foreach (utils::Matrix input, std::function<double (double)> func) {
    std::transform(input.data().begin(), input.data().end(), input.data().begin(), std::move(func));
    return input;
}
