//
// Created by Esteban on 6/16/2022.
//

#ifndef UNTITLED21_MLPCLASSIFIER_H
#define UNTITLED21_MLPCLASSIFIER_H

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


class MLPClassifier : public Network {
public:
    using Matrix = mlp::math::Matrix;
    using Settings = NetworkSettings;

    template<class Act>
    struct Activation {};

    using Shape = std::vector<size_t>;

public:
    template<class Act>
    MLPClassifier (const Shape& sizes, Activation<Act>, const Settings& settings = Settings());

    double calculate_loss (const Matrix& real_value) const;

    Matrix forward (const Matrix& input) override;

    void backward (const Matrix& real_value) override;

    NetworkMetrics metrics (const Matrix& input, const Matrix& output);

    const char* act_name () const;

public:
    std::function<void (MLPClassifier*, size_t)> on_epoch_callback_;

private:
    void on_epoch_callback (size_t epoch) override;

    double loss = NAN;
    std::vector<DenseLayer> dense;
    std::vector<std::unique_ptr<ActivationLayer>> act;
    SoftmaxLoss softmax;
};

template<class Act>
MLPClassifier::MLPClassifier (const std::vector<size_t>& sizes, Activation<Act>, const NetworkSettings& settings)
    : Network(settings) {
    dense.emplace_back(sizes[0], sizes[1]);
    for (size_t i = 1; i < sizes.size() - 1; i++) {
        act.emplace_back(std::make_unique<Act>());
        dense.emplace_back(sizes[i], sizes[i + 1]);
    }
    std::cout << "NetworkReLU: " << sizes.size() << std::endl;
}

#endif //UNTITLED21_MLPCLASSIFIER_H
