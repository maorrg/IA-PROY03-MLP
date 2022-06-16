//
// Created by Esteban on 6/4/2022.
//

#ifndef UNTITLED21_DENSELAYER_H
#define UNTITLED21_DENSELAYER_H

#include <mlp/net/Layer.h>

class DenseLayer : public Layer {
public:
    using Matrix = Layer::Matrix;

public:
    DenseLayer (size_t input_size, size_t output_size);

    Matrix forward (const Matrix& input_) override;

    Matrix backward (const Matrix& gradient, double learning_rate) override;

protected:
    Matrix weights;
    Matrix biases;
    Matrix input;
};


#endif //UNTITLED21_DENSELAYER_H
