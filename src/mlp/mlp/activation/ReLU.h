//
// Created by Esteban on 6/5/2022.
//

#ifndef UNTITLED21_RELU_H
#define UNTITLED21_RELU_H

#include <mlp/activation/ActivationLayer.h>

class ReLU : public ActivationLayer {
public:
    using Matrix = ActivationLayer::Matrix;
    using Function = std::function<Matrix (const Matrix&)>;
public:
    ReLU ();
private:
    static Matrix relu (const Matrix& input_);
    static Matrix relu_derivative (const Matrix& input_);
};


#endif //UNTITLED21_RELU_H
