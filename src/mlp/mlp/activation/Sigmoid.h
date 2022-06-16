//
// Created by Esteban on 6/5/2022.
//

#ifndef UNTITLED21_SIGMOID_H
#define UNTITLED21_SIGMOID_H


#include <mlp/Layer.h>

class Sigmoid : public Layer {
public:
    using Matrix = Layer::Matrix;
public:
    Sigmoid () = default;
    Matrix forward (const Matrix& input_) override;
    Matrix backward (const Matrix& gradient, double learning_rate) override;
private:
    static Matrix sigmoid (const Matrix& input_);
    static Matrix sigmoid_derivative (const Matrix& input_);

    Matrix output;
};


#endif //UNTITLED21_SIGMOID_H
