//
// Created by Esteban on 6/5/2022.
//

#ifndef UNTITLED21_MSE_H
#define UNTITLED21_MSE_H

#include "LossFunction.h"

class MSE : public LossFunction {
public:
    MSE();
    static double mse_loss(const Matrix& output, const Matrix& target);
    static Matrix mse_derivative(const Matrix& output, const Matrix& target);
};



#endif //UNTITLED21_MSE_H
