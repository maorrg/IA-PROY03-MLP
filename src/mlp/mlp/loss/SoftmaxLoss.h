//
// Created by Esteban on 6/7/2022.
//

#ifndef UNTITLED21_SOFTMAXLOSS_H
#define UNTITLED21_SOFTMAXLOSS_H

#include "LossLayer.h"

class SoftmaxLoss : public LossLayer {
public:
    using Matrix = boost::numeric::ublas::matrix<double>;

public:
    explicit SoftmaxLoss (double epsilon = 1e-8);

    Matrix forward (const Matrix& input_) override;

    Matrix backward (const Matrix& real_value_, double learning_rate) override;

    [[nodiscard]]
    double loss (const Matrix& real_value_) const override;

private:
    Matrix output;
    double epsilon;
};


#endif //UNTITLED21_SOFTMAXLOSS_H
