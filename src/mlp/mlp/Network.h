//
// Created by Esteban on 6/4/2022.
//

#ifndef UNTITLED21_NETWORK_H
#define UNTITLED21_NETWORK_H

#include <mlp/math/boost_math.h>
#include <random>

inline auto default_random () {
    static auto random_source = std::random_device{};
    static auto random_engine = std::default_random_engine{ random_source() };
    static auto dist = std::uniform_real_distribution<double>(-0.5, 0.5);
    return dist(random_engine);
};

struct NetworkSettings {
    size_t epochs = 100;
    size_t batch_size = 64;
    double learning_rate = 0.01;
    double (*random) () = default_random;
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
