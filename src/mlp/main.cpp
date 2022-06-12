#include "mlp/Network.h"
#include "mlp/net/nets.h"
#include "mlp/activation/activations.h"
#include "mlp/loss/losses.h"
#include <iostream>
#include <random>

int main () {
    namespace ub = boost::numeric::ublas;
    using Matrix = ub::matrix<double>;
    static auto random_source = std::random_device{};
    static auto random_engine = std::default_random_engine{random_source()};
    static auto dist = std::uniform_real_distribution<double>(-0.5, 0.5);
    const auto random = [&]() { return dist(random_engine); };

    auto L1 = DenseLayer(2, 8, random);
    auto A1 = ReLU();
    auto L2 = DenseLayer(8, 8, random);
    auto A2 = ReLU();
    auto L3 = DenseLayer(8, 2, random);
    auto loss = SoftmaxLoss();

    Network network({ &L1, &A1, &L2, &A2, &L3 }, &loss);

    auto input = Matrix{2, 4};
    auto output = Matrix{2, 4};
    input(0, 0) = 1; input(1, 0) = 1; output(0, 0) = 1; output(1, 0) = 0;
    input(0, 1) = 0; input(1, 1) = 1; output(0, 1) = 0; output(1, 1) = 1;
    input(0, 2) = 1; input(1, 2) = 0; output(0, 2) = 0; output(1, 2) = 1;
    input(0, 3) = 0; input(1, 3) = 0; output(0, 3) = 1; output(1, 3) = 0;

    network.train(input, output, 1000, 0.1);
    std::cout << network.forward(input) << std::endl;
}