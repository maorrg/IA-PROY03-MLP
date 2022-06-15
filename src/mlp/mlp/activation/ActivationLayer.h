//
// Created by Esteban on 6/4/2022.
//

#ifndef UNTITLED21_ACTIVATIONLAYER_H
#define UNTITLED21_ACTIVATIONLAYER_H

#include <mlp/Layer.h>

class ActivationLayer : public Layer {
public:
    using Matrix = Layer::Matrix;
    using Function = std::function<Matrix (const Matrix&)>;
public:
    ActivationLayer (Function function_, Function derivative_);
    Matrix forward (const Matrix& input_) override;
    Matrix backward (const Matrix& gradient, double) override;

private:
    Function function;
    Function derivative;
    Matrix input;
};


#endif //UNTITLED21_ACTIVATIONLAYER_H
