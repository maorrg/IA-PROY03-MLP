//
// Created by Esteban on 6/4/2022.
//

#ifndef UNTITLED21_TANH_H
#define UNTITLED21_TANH_H

#include <mlp/net/Layer.h>
#include "ActivationLayer.h"


class Tanh : public ActivationLayer {
public:
    using Matrix = Layer::Matrix;
public:
    Tanh () = default;
    Matrix forward (const Matrix& input_) override;
    Matrix backward (const Matrix& gradient, double learning_rate) override;
    const char* name() const override;
private:
    static Matrix tanh (const Matrix& input_);
    static Matrix tanh_derivative (const Matrix& input_);

    Matrix output;
};


#endif //UNTITLED21_TANH_H
