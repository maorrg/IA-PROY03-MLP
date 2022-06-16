//
// Created by Esteban on 6/5/2022.
//

#ifndef UNTITLED21_SIGMOID_H
#define UNTITLED21_SIGMOID_H


#include <mlp/activation/ActivationLayer.h>

class Sigmoid : public ActivationLayer {
public:
    using Matrix = ActivationLayer::Matrix;
public:
    Sigmoid ();
private:
    static Matrix sigmoid (const Matrix& input_);
    static Matrix sigmoid_derivative (const Matrix& input_);
};


#endif //UNTITLED21_SIGMOID_H
