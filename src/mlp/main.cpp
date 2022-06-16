
#include <mlp/Network.h>
#include <mlp/net/nets.h>
#include <mlp/activation/activations.h>
#include <mlp/loss/losses.h>
#include <iostream>
#include <random>


class MyNetwork : public Network<MyNetwork> {
public:
    using Matrix = mlp::math::Matrix;

    MyNetwork (size_t input_size, size_t output_size, const NetworkSettings& settings)
        : Network(settings),
          dense1(input_size, 8),
          dense2(8, 8),
          dense3(8, output_size) {
    }

    Matrix forward (const Matrix& input) override {
        auto x = dense1.forward(input);
        x = relu1.forward(x);
        x = dense2.forward(x);
        x = relu2.forward(x);
        x = dense3.forward(x);
        x = softmax.forward(x);
        return x;
    }

    void backward (const Matrix& real_value) override {
        loss = softmax.loss(real_value);
        auto x = softmax.backward(real_value, settings.learning_rate);
        x = dense3.backward(x, settings.learning_rate);
        x = relu2.backward(x, settings.learning_rate);
        x = dense2.backward(x, settings.learning_rate);
        x = relu1.backward(x, settings.learning_rate);
        dense1.backward(x, settings.learning_rate);
    }

protected:
    void on_epoch_callback (size_t epoch) override {
        std::cout << "Epoch: " << epoch << ' ' << "Loss: " << loss << std::endl;
    }

private:
    DenseLayer dense1;
    ReLU relu1;
    DenseLayer dense2;
    ReLU relu2;
    DenseLayer dense3;
    SoftmaxLoss softmax;
    double loss = NAN;
};

int main () {
    using Matrix = mlp::math::Matrix;

    mlp::math::setDevice(0);
    mlp::math::info();

    Matrix input = transpose(join(1, tile(af::seq(0, 1), 2), flat( transpose(tile(af::seq(0, 1), 1, 2)))));

    Matrix output = mlp::math::constant(0, 2, 4);

    output(0, mlp::math::span) = (input(0, mlp::math::span) + input(1, mlp::math::span)) % 2;
    output(1, mlp::math::span) = 1 - output(0, mlp::math::span);
    auto network = MyNetwork(2, 2, {1000, 0.1});
    network.train(input, output);

    Matrix predicted = network.forward(input);
    std::cout << "Input: " << input << std::endl;
    std::cout << "Output: " << output << std::endl;
    std::cout << "Predicted: " << predicted << std::endl;

    return 0;
}