#include "experiment/NetworkReLU.h"
#include "experiment/input.h"
#include "experiment/train_test_split.h"
#undef max
#undef min

int main () {
    using namespace mlp::math;
    std::ios_base::sync_with_stdio(false);

    mlp::math::setDevice(0);
    mlp::math::info();
//    mlp::math::setSeed(42);


    std::cout << "Loading dataset..." << std::endl;
    auto Xy = load_dataset("../../data/output/imageProcessingOutput/eigenvectors.csv");
    std::cout << "Dataset loaded. Processing..." << std::endl;

    auto Xy_split = split_dataset(Xy, 0.7);

    auto Xy_train = Xy_split.train, Xy_test = Xy_split.test;
    auto X_train = Xy_train.data, y_train = Xy_train.labels;
    auto X_test = Xy_test.data, y_test = Xy_test.labels;
    auto y_labels_train = y_train, y_labels_test = y_test;
    y_train = to_categorical(y_train, 10), y_test = to_categorical(y_test, 10);

    X_train = mlp::math::transpose(X_train);
    X_test = mlp::math::transpose(X_test);
    y_train = mlp::math::transpose(y_train);
    y_test = mlp::math::transpose(y_test);

    std::cout << "X_train: " << mlp::math::size(X_train, 0) << "x" << mlp::math::size(X_train, 1) << std::endl;
    std::cout << "y_train: " << mlp::math::size(y_train, 0) << "x" << mlp::math::size(y_train, 1) << std::endl;
    std::cout << "X_test: " << mlp::math::size(X_test, 0) << "x" << mlp::math::size(X_test, 1) << std::endl;
    std::cout << "y_test: " << mlp::math::size(y_test, 0) << "x" << mlp::math::size(y_test, 1) << std::endl;
    std::cout << "Labels train: " << mlp::math::size(y_labels_train, 0) << "x" << mlp::math::size(y_labels_train, 1) << std::endl;
    std::cout << "Labels test: " << mlp::math::size(y_labels_test, 0) << "x" << mlp::math::size(y_labels_test, 1) << std::endl;

    auto model = NetworkReLU(
        mlp::math::size(X_train, 0),
        mlp::math::size(y_train, 0),
        NetworkSettings{60, 0.1}
    );
    model.on_epoch_callback_ = [=](NetworkReLU* self, const size_t epoch) {
        std::cout << "=========== Epoch: " << epoch << "================" << std::endl;
        std::cout << "Loss train: " << self->loss << std::endl;
        self->forward(X_test);
        auto test_loss = self->calculate_loss(y_test);
        std::cout << "Loss test: " << test_loss << std::endl;
    };

    model.train(X_train, y_train);
    return 0;
}
