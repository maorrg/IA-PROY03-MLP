#include "TanhLayer.h"

TanhLayer::TanhLayer () : ActivationLayer(TanhLayer::tanh, TanhLayer::tanh_derivative) {}

TanhLayer::Matrix TanhLayer::tanh (const TanhLayer::Matrix& input_) {
    Matrix output = input_;
    std::transform(output.data().begin(), output.data().end(), output.data().begin(), [](double x) {
        return std::tanh(x);
    });
    return output;
}

TanhLayer::Matrix TanhLayer::tanh_derivative (const TanhLayer::Matrix& input_) {
    Matrix output = input_;
    std::transform(output.data().begin(), output.data().end(), output.data().begin(), [](double x) {
        return 1 - std::pow(std::tanh(x), 2);
    });
    return output;
}
