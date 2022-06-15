//
// Created by Esteban on 6/4/2022.
//

#ifndef UNTITLED21_TANH_H
#define UNTITLED21_TANH_H

#include <mlp/activation/ActivationLayer.h>

class Tanh : public ActivationLayer {
public:
    using Matrix = ActivationLayer::Matrix;
    using Function = std::function<Matrix (const Matrix&)>;
public:
    Tanh ();
private:
    static Matrix tanh (const Matrix& input_);
    static Matrix tanh_derivative (const Matrix& input_);
};


#endif //UNTITLED21_TANH_H
