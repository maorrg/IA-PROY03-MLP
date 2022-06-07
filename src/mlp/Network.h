//
// Created by Esteban on 6/4/2022.
//

#ifndef UNTITLED21_NETWORK_H
#define UNTITLED21_NETWORK_H

#include <boost/numeric/ublas/io.hpp>
#include <utility>

#include "layer/Layer.h"
#include "loss/LossFunction.h"

class Network {
public:
    using Matrix = boost::numeric::ublas::matrix<double>;

public:
    explicit Network(std::vector<Layer*> layers_, LossFunction lossFunction_);

    Network& train(const Matrix& input, const Matrix& target, size_t epochs, double learning_rate);

    Matrix predict(const Matrix& input);

private:
    // non owning
    std::vector<Layer*> layers;
    LossFunction lossFunction;
};

#endif  // UNTITLED21_NETWORK_H
