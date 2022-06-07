//
// Created by Esteban on 6/4/2022.
//

#ifndef UNTITLED21_TANHLAYER_H
#define UNTITLED21_TANHLAYER_H

#include "ActivationLayer.h"

class TanhLayer : public ActivationLayer {
public:
    using Matrix = boost::numeric::ublas::matrix<double>;
    using Function = std::function<Matrix (const Matrix&)>;
public:
    TanhLayer ();
private:
    static Matrix tanh (const Matrix& input_);
    static Matrix tanh_derivative (const Matrix& input_);
};


#endif //UNTITLED21_TANHLAYER_H
