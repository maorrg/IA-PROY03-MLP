//
// Created by Esteban on 6/4/2022.
//

#ifndef UNTITLED21_NETWORK_H
#define UNTITLED21_NETWORK_H

#include "mlp/math/mlp_math.h"
#include <random>


struct NetworkSettings {
    size_t epochs = 100;
    double learning_rate = 0.01;
};

template<class NetworkType>
class Network {
public:
    using Matrix = mlp::math::Matrix;

public:
    explicit Network (const NetworkSettings& settings);
    NetworkType& train (const Matrix& input, const Matrix& target);

    virtual Matrix forward (const Matrix& input) = 0;
    virtual void backward (const Matrix& real_value) = 0;

protected:
    virtual void on_epoch_callback (size_t epoch);

protected:
    NetworkSettings settings;
};


template<class NetworkType>
Network<NetworkType>::Network (const NetworkSettings& settings) : settings(settings) {}

template<class NetworkType>
NetworkType& Network<NetworkType>::train (const Network::Matrix& input, const Network::Matrix& target) {
    for (size_t epoch = 0; epoch < settings.epochs; ++epoch) {
        this->forward(input);
        this->backward(target);
        this->on_epoch_callback(epoch);
    }
    return *static_cast<NetworkType*>(this);
}

template<class NetworkType>
void Network<NetworkType>::on_epoch_callback (size_t) {}


#endif  // UNTITLED21_NETWORK_H
