//
// Created by Esteban on 6/7/2022.
//

#ifndef UNTITLED21_MSELOSS_H
#define UNTITLED21_MSELOSS_H

#include "LossLayer.h"

class MSELoss : public LossLayer {
public:
    using Matrix = LossLayer::Matrix;

public:
    MSELoss () = default;

    Matrix forward (const Matrix& input_) override;

    Matrix backward (const Matrix& real_value_, double learning_rate) override;

    [[nodiscard]]
    double loss (const Matrix& real_value_) const override;

private:
    Matrix output;
};


#endif //UNTITLED21_MSELOSS_H
