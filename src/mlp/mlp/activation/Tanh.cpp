#include "Tanh.h"

Tanh::Tanh () : ActivationLayer(Tanh::tanh, Tanh::tanh_derivative) {}

Tanh::Matrix Tanh::tanh (const Tanh::Matrix& input_) {
    Matrix output = input_;
    std::transform(output.data().begin(), output.data().end(), output.data().begin(), [](double x) {
        return std::tanh(x);
    });
    return output;
}

Tanh::Matrix Tanh::tanh_derivative (const Tanh::Matrix& input_) {
    Matrix output = input_;
    std::transform(output.data().begin(), output.data().end(), output.data().begin(), [](double x) {
        return 1 - std::pow(std::tanh(x), 2);
    });
    return output;
}
