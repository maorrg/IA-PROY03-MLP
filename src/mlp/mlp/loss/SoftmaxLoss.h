//
// Created by Esteban on 6/7/2022.
//

#ifndef UNTITLED21_SOFTMAXLOSS_H
#define UNTITLED21_SOFTMAXLOSS_H

#include "LossLayer.h"

class SoftmaxLoss : public LossLayer {
public:
    using Matrix = LossLayer::Matrix;

public:
    SoftmaxLoss () = default;

    Matrix forward (const Matrix& input_) override;

    Matrix backward (const Matrix& real_value_, double learning_rate) override;

    [[nodiscard]]
    double loss (const Matrix& real_value_) const override;

protected:
    Matrix output;
};


#endif //UNTITLED21_SOFTMAXLOSS_H
