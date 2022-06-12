//
// Created by Esteban on 6/4/2022.
//

#ifndef UNTITLED21_DENSELAYER_H
#define UNTITLED21_DENSELAYER_H

#include "../Layer.h"

class DenseLayer : public Layer {
public:
    using Matrix = boost::numeric::ublas::matrix<double>;
    using Random = std::function<double (void)>;

public:
    DenseLayer (size_t input_size, size_t output_size);

    DenseLayer (size_t input_size, size_t output_size, const Random& random);

    Matrix forward (const Matrix& input_) override;

    Matrix backward (const Matrix& gradient, double learning_rate) override;

protected:
    Matrix weights;
    Matrix biases;
    Matrix input;
};


#endif //UNTITLED21_DENSELAYER_H
