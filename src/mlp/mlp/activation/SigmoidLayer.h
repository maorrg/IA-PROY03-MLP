//
// Created by Esteban on 6/5/2022.
//

#ifndef UNTITLED21_SIGMOIDLAYER_H
#define UNTITLED21_SIGMOIDLAYER_H


#include <mlp/activation/ActivationLayer.h>

class SigmoidLayer : public ActivationLayer {
public:
    using Matrix = ActivationLayer::Matrix;
    using Function = std::function<Matrix (const Matrix&)>;

public:
    SigmoidLayer ();
private:
    static Matrix sigmoid (const Matrix& input_);
    static Matrix sigmoid_derivative (const Matrix& input_);
};


#endif //UNTITLED21_SIGMOIDLAYER_H
