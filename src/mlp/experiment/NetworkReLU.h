//
// Created by Esteban on 6/16/2022.
//

#ifndef UNTITLED21_NETWORKRELU_H
#define UNTITLED21_NETWORKRELU_H

#include <mlp/Network.h>
#include <mlp/net/nets.h>
#include <mlp/activation/activations.h>
#include <mlp/loss/losses.h>

class NetworkReLU : public Network<NetworkReLU> {
public:
    using Matrix = mlp::math::Matrix;

    NetworkReLU (size_t input_size, size_t output_size, const NetworkSettings& settings);

    Matrix forward (const Matrix& input) override;

    void backward (const Matrix& real_value) override;

protected:
    void on_epoch_callback (size_t epoch) override;

private:
    DenseLayer dense1;
    ReLU relu1;
    DenseLayer dense2;
    ReLU relu2;
    DenseLayer dense3;
    SoftmaxLoss softmax;
    double loss = NAN;
};


#endif //UNTITLED21_NETWORKRELU_H
