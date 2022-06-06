//
// Created by Esteban on 6/5/2022.
//

#ifndef UNTITLED21_RELULAYER_H
#define UNTITLED21_RELULAYER_H

#include "ActivationLayer.h"

class ReluLayer : public ActivationLayer {
public:
    using Matrix = boost::numeric::ublas::matrix<double>;
    using Function = std::function<Matrix (const Matrix&)>;
public:
    ReluLayer ();
private:
    static Matrix relu (const Matrix& input_);
    static Matrix relu_derivative (const Matrix& input_);
};


#endif //UNTITLED21_RELULAYER_H
