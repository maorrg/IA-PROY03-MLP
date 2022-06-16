//
// Created by Esteban on 6/15/2022.
//

#ifndef UNTITLED21_AF_BACKEND_H
#define UNTITLED21_AF_BACKEND_H

#include <arrayfire.h>
#include <iostream>

namespace mlp::math {
    using Matrix = af::array;
    using namespace af;

    void set_seed(unsigned long long seed);
    Matrix zeros(size_t n, size_t m);
}

std::ostream& operator<<(std::ostream& os, const mlp::math::Matrix & mat);

#define MLP_MATH_MAKE_FUNCTOR(NAME, OP)



#endif //UNTITLED21_AF_BACKEND_H
