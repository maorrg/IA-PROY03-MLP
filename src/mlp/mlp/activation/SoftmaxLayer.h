//
// Created by Esteban on 6/4/2022.
//

#ifndef UNTITLED21_SOFTMAX_LAYER_H
#define UNTITLED21_SOFTMAX_LAYER_H

#include "../Layer.h"

class SoftmaxLayer : public Layer {
public:
    using Matrix = boost::numeric::ublas::matrix<double>;

public:
    Matrix forward (const Matrix& input_) override;
    Matrix backward (const Matrix& gradient, double learning_rate) override;

private:
    Matrix output;
};


#endif //UNTITLED21_SOFTMAX_LAYER_H
