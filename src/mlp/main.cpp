

#include "mlp/Network.h"
#include "mlp/net/nets.h"
#include "mlp/activation/activations.h"
#include "mlp/loss/losses.h"
#include <iostream>
#include <random>

int main () {
    using namespace boost::numeric::ublas;
    static auto random_engine = std::default_random_engine{42};
    static auto dist = std::uniform_real_distribution<double>(-0.5, 0.5);
    const auto random = [&]() { return dist(random_engine); };

    DenseLayer L1(2, 8, random);
    ReluLayer L2;
    DenseLayer L3(8, 8, random);
    ReluLayer L4;
    DenseLayer L5(8, 2, random);
    SoftmaxLoss loss_function;

    Network network({ &L1, &L2, &L3, &L4, &L5 }, &loss_function);
    
    DenseLayer::Matrix input{2, 4};
    DenseLayer::Matrix output{2, 4};
    input(0, 0) = 1; input(1, 0) = 1; output(0, 0) = 1; output(1, 0) = 0;
    input(0, 1) = 0; input(1, 1) = 1; output(0, 1) = 0; output(1, 1) = 1;
    input(0, 2) = 1; input(1, 2) = 0; output(0, 2) = 0; output(1, 2) = 1;
    input(0, 3) = 0; input(1, 3) = 0; output(0, 3) = 1; output(1, 3) = 0;

    network.train(input, output, 2000, 0.1);
    std::cout << network.forward(input) << std::endl;
}