//
// Created by Esteban on 6/5/2022.
//

#ifndef UNTITLED21_CROSSENTROPY_H
#define UNTITLED21_CROSSENTROPY_H

#include "LossFunction.h"

class CrossEntropy : public LossFunction {
public:
    CrossEntropy();
    static double cross_entropy_loss(const Matrix& output, const Matrix& target);
    static Matrix cross_entropy_derivative(const Matrix& output, const Matrix& target);
};


#endif //UNTITLED21_CROSSENTROPY_H
