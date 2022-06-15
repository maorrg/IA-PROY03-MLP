#include "mlp/Network.h"
#include "mlp/net/nets.h"
#include "mlp/activation/activations.h"
#include "mlp/loss/losses.h"
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/assignment.hpp>
#include <random>

class MyNetwork : public Network<MyNetwork> {
public:
    using Matrix = mlp::math::Matrix;

    MyNetwork (size_t input_size, size_t output_size, const NetworkSettings& settings)
        : Network(settings),
          dense1(input_size, 8, settings.random),
          dense2(8, 8, settings.random),
          dense3(8, output_size, settings.random) {

    }

    Matrix forward (const Matrix& input) override {
        Matrix x = dense1.forward(input);
        x = relu1.forward(x);
        x = dense2.forward(x);
        x = relu2.forward(x);
        x = dense3.forward(x);
        x = softmax.forward(x);
        return x;
    }

    void backward (const Matrix& real_value) override {
        loss = softmax.loss(real_value);

        Matrix x = softmax.backward(real_value, settings.learning_rate);
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

    auto input = Matrix{ 2, 4 };
    auto output = Matrix{ 2, 4 };

    input <<=
        1, 0, 1, 0,
        1, 1, 0, 0;

    output <<=
        1, 0, 0, 1,
        0, 1, 1, 0;


    MyNetwork network(2, 2, { 1000, 4, 0.1 });
    network.train(input, output);
    std::cout << network.forward(input) << '\n';

    return 0;
}