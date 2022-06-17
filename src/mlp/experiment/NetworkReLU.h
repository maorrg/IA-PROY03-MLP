//
// Created by Esteban on 6/16/2022.
//

#ifndef UNTITLED21_NETWORKRELU_H
#define UNTITLED21_NETWORKRELU_H

#include <mlp/net/Network.h>
#include <functional>

class NetworkReLU : public Network {
public:
    using Matrix = mlp::math::Matrix;

    NetworkReLU (size_t input_size, size_t output_size, const NetworkSettings& settings);

    Matrix forward (const Matrix& input) override;

    void backward (const Matrix& real_value) override;

    double calculate_loss (const Matrix& real_value) const;

public:
    std::function<void(NetworkReLU*, size_t)> on_epoch_callback_;
    double loss = NAN;

    void on_epoch_callback (size_t epoch) override;

    DenseLayer dense1;
    ReLU relu1;
    DenseLayer dense2;
    ReLU relu2;
    DenseLayer dense3;
    SoftmaxLoss softmax;
};


#endif //UNTITLED21_NETWORKRELU_H
