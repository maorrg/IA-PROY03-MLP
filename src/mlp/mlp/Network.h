//
// Created by Esteban on 6/4/2022.
//

#ifndef UNTITLED21_NETWORK_H
#define UNTITLED21_NETWORK_H

#include <boost/numeric/ublas/io.hpp>
#include <utility>

#include "Layer.h"
#include "loss/LossLayer.h"

class Network {
public:
    using Matrix = boost::numeric::ublas::matrix<double>;

public:
    explicit Network(std::vector<Layer*> hidden_layers_, LossLayer* loss_layer_);

    Network& train(const Matrix& input, const Matrix& target, size_t epochs, double learning_rate);

    Matrix forward(const Matrix& input);

    void backward(const Matrix& real_value, double learning_rate);

private:
    // non owning
    std::vector<Layer*> hidden_layers;
    LossLayer* loss_layer;
};

#endif  // UNTITLED21_NETWORK_H
