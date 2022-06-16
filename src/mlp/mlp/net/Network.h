//
// Created by Esteban on 6/4/2022.
//

#ifndef UNTITLED21_NETWORK_H
#define UNTITLED21_NETWORK_H

#include <mlp/math/mlp_math.h>
#include <mlp/net/connected.h>
#include <mlp/net/activations.h>
#include <mlp/net/losses.h>
#include <random>


struct NetworkSettings {
    size_t epochs = 100;
    double learning_rate = 0.01;
};

class Network {
public:
    using Matrix = mlp::math::Matrix;

public:
    explicit Network (const NetworkSettings& settings);
    void train (const Matrix& input, const Matrix& target);

    virtual Matrix forward (const Matrix& input) = 0;
    virtual void backward (const Matrix& real_value) = 0;

protected:
    virtual void on_epoch_callback (size_t epoch);

protected:
    NetworkSettings settings;
};




#endif  // UNTITLED21_NETWORK_H
