

#include <iostream>
#include "experiment/NetworkReLU.h"

int main () {
    using Matrix = mlp::math::Matrix;

    mlp::math::setDevice(0);
    mlp::math::info();

    Matrix input = mlp::math::zeros(2, 4);
    Matrix output = mlp::math::zeros(2, 4);
    input(0, 0) = 1; input(1, 0) = 1; output(0, 0) = 1; output(1, 0) = 0;
    input(0, 1) = 0; input(1, 1) = 1; output(0, 1) = 0; output(1, 1) = 1;
    input(0, 2) = 1; input(1, 2) = 0; output(0, 2) = 0; output(1, 2) = 1;
    input(0, 3) = 0; input(1, 3) = 0; output(0, 3) = 1; output(1, 3) = 0;

    auto network = NetworkReLU(2, 2, {1000, 0.1});
    network.train(input, output);

    Matrix predicted = network.forward(input);

    std::cout << "Input: " << input << std::endl;
    std::cout << "Output: " << output << std::endl;
    std::cout << "Predicted: " << predicted << std::endl;

    return 0;
}