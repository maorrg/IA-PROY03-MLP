#include "experiment/dataset_utils.h"
#include "experiment/MLPClassifier.h"
#include <experiment/Teebuf.h>
#include <fstream>

using namespace mlp::math;

void run_experiment (
    MLPClassifier& clf,
    const Dataset::Matrix& X_train,
    const Dataset::Matrix& y_train,
    const Dataset::Matrix& X_test,
    const Dataset::Matrix& y_test,
    const std::string& path
) {
    std::ofstream out(path, std::ios::out);
    teestream tee(std::cout, out);

    const auto callback = [&] (MLPClassifier* self, const size_t epoch) {
        auto metrics = self->metrics(X_test, y_test);
        tee << epoch << "," << self->act_name() << ","
            << metrics.train_loss << "," << metrics.test_loss << ","
            << metrics.accuracy << "," << metrics.precision << ","
            << metrics.recall << "," << metrics.f1_score << std::endl;
    };

    tee << "epoch,activation_function,train_loss,test_loss,accuracy,precision,recall,f1_score" << std::endl;
    clf.on_epoch_callback_ = callback;
    clf.train(X_train, y_train);
}

void run_layer_experiments (
    const Dataset::Matrix& X_train,
    const Dataset::Matrix& y_train,
    const Dataset::Matrix& X_test,
    const Dataset::Matrix& y_test
) {
    size_t in = size(X_train, 0), out = size(y_train, 0);
    auto relu_model = MLPClassifier({ in, 100, out }, MLPClassifier::Activation<ReLU>{});
    auto sigmoid_model = MLPClassifier({ in, 100, out }, MLPClassifier::Activation<Sigmoid>{});
    auto tanh_model = MLPClassifier({ in, 100, out }, MLPClassifier::Activation<Tanh>{});

//    run_experiment(relu_model, X_train, y_train, X_test, y_test, "../data/relu_model.csv");
    run_experiment(sigmoid_model, X_train, y_train, X_test, y_test, "../data/sigmoid_model.csv");
    run_experiment(tanh_model, X_train, y_train, X_test, y_test, "../data/tanh_model.csv");
}

void run_shape_experiments (
    const Dataset::Matrix& X_train,
    const Dataset::Matrix& y_train,
    const Dataset::Matrix& X_test,
    const Dataset::Matrix& y_test
) {
    size_t in = size(X_train, 0), out = size(y_train, 0);
    for (size_t height : { 8, 16, 32, 64, 128 }) {
        for (int depth : { 1, 2, 3, 4 }) {
            std::string name = "../data/shape/relu_" + std::to_string(height) + "_" + std::to_string(depth) + ".csv";

            MLPClassifier::Shape sizes = { in };
            for (int i = 0; i < depth; i++) sizes.push_back(height);
            sizes.push_back(out);

            auto model = MLPClassifier(sizes, MLPClassifier::Activation<ReLU>{});
            run_experiment(model, X_train, y_train, X_test, y_test, name);
        }
    }
}

int main () {
    std::ios_base::sync_with_stdio(false);

    setDevice(0);
    info();
//    setSeed(42);

    auto Xy = load_dataset("../../data/output/imageProcessingOutput/eigenvectors.csv", "../Xy.cache", true);
    auto X = Xy.data, y = Xy.labels;

    auto[Xy_train, Xy_test] = train_test_split(Xy, 0.7);
    auto X_train = Xy_train.data, y_train = Xy_train.labels, X_test = Xy_test.data, y_test = Xy_test.labels;

    for (auto* x : { &X, &y, &X_train, &y_train, &X_test, &y_test }) {
        *x = transpose(*x);
    }

    run_layer_experiments(X_train, y_train, X_test, y_test);
    run_shape_experiments(X_train, y_train, X_test, y_test);

    return 0;
}
