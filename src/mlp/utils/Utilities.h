#include <utility>

//
// Created by Esteban on 6/5/2022.
//

#ifndef UNTITLED21_UTILITIES_H
#define UNTITLED21_UTILITIES_H

#include "boost/numeric/ublas/matrix.hpp"

namespace utils {
    using Matrix = boost::numeric::ublas::matrix<double>;

    Matrix ones (size_t size_1, size_t size_2);;

    Matrix foreach (Matrix input, std::function<double(double)> func);
}

#endif //UNTITLED21_UTILITIES_H
