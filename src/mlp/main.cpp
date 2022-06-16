#include "experiment/NetworkReLU.h"
#include "experiment/input.h"

int main () {
    using namespace mlp::math;

    mlp::math::setDevice(0);
    mlp::math::info();

    auto [X, Y] = load_dataset("../data/output/imageProcessingOutput/eigenvectors.csv");
    NetworkReLU network(size(X, 0), size(Y, 0), {5000, 0.1});
    network.train(X, Y);

    return 0;
}
