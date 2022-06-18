//
// Created by Esteban on 6/16/2022.
//

#ifndef UNTITLED21_NETWORKRELU_H
#define UNTITLED21_NETWORKRELU_H

#include <mlp/net/Network.h>
#include <functional>

struct NetworkMetrics {
    //    epoch,activation_function,train_loss,test_loss,accuracy,precision,recall,f1_score
    double train_loss;
    double test_loss;
    double accuracy;
    double precision;
    double recall;
    double f1_score;
};

class NetworkReLU : public Network {
public:
    using Matrix = mlp::math::Matrix;

public:
    NetworkReLU (const std::vector<size_t>& sizes, const NetworkSettings& settings);
    double calculate_loss (const Matrix& real_value) const;
    Matrix forward (const Matrix& input) override;
    void backward (const Matrix& real_value) override;
    NetworkMetrics metrics(const Matrix& input, const Matrix& output);

    const char* act_name() const;

public:
    std::function<void(NetworkReLU*, size_t)> on_epoch_callback_;

private:
    void on_epoch_callback (size_t epoch) override;

    double loss = NAN;
    std::vector<DenseLayer> dense;
    std::vector<std::unique_ptr<Activation>> act;
    SoftmaxLoss softmax;
};


#endif //UNTITLED21_NETWORKRELU_H
