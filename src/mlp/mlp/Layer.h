//
// Created by Esteban on 6/4/2022.
//

#ifndef UNTITLED21_LAYER_H
#define UNTITLED21_LAYER_H

#include <mlp/math/boost_math.h>

class Layer {
public:
    using Matrix = mlp::math::Matrix;

public:
    Layer () = default;
    virtual ~Layer () = default;

    virtual Matrix forward (const Matrix& input) = 0;
    virtual Matrix backward (const Matrix& gradient, double learning_rate) = 0;
};


#endif //UNTITLED21_LAYER_H
