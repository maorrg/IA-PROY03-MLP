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

    static af::randomEngine engine = af::getDefaultRandomEngine();

    inline void set_seed(unsigned long long seed) {
        engine.setSeed(seed);
    }
}

inline std::ostream& operator<<(std::ostream& os, const mlp::math::Matrix & mat) {
    const char* res = af::toString("", mat);
    os << res;
    af::freeHost(res);
    return os;
}

#define MLP_MATH_MAKE_FUNCTOR(NAME, OP)



#endif //UNTITLED21_AF_BACKEND_H
