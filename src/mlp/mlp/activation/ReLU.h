//
// Created by Esteban on 6/5/2022.
//

#ifndef UNTITLED21_RELU_H
#define UNTITLED21_RELU_H

#include <mlp/net/Layer.h>

class ReLU : public Layer {
public:
    using Matrix = Layer::Matrix;
public:
    ReLU () = default;
    Matrix forward (const Matrix& input_) override;
    Matrix backward (const Matrix& gradient, double learning_rate) override;
private:
    static Matrix relu (const Matrix& input_);
    static Matrix relu_derivative (const Matrix& input_);

    Matrix input;
};


#endif //UNTITLED21_RELU_H
