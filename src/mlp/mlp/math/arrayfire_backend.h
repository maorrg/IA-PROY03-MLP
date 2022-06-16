//
// Created by Esteban on 6/15/2022.
//

#ifndef UNTITLED21_AF_BACKEND_H
#define UNTITLED21_AF_BACKEND_H

#include "pragma.h"

DISABLE_WARNING_PUSH
#if defined(_MSC_VER)
DISABLE_WARNING(4275)
DISABLE_WARNING(4201)
#endif
#include <arrayfire.h>
DISABLE_WARNING_POP

#include <iostream>

namespace mlp::math {
    using Matrix = af::array;
    using namespace af;

    void setSeed(unsigned long long seed);
    Matrix zeros(size_t n, size_t m);
    size_t size(const Matrix& mat, size_t dim);
}

std::ostream& operator<<(std::ostream& os, const mlp::math::Matrix & mat);

#define MLP_MATH_MAKE_FUNCTOR(NAME, OP)



#endif //UNTITLED21_AF_BACKEND_H
