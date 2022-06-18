//
// Created by Esteban on 6/16/2022.
//

#include "MLPClassifier.h"



MLPClassifier::Matrix MLPClassifier::forward (const MLPClassifier::Matrix& input) {
//    auto x = dense1.forward(input);
//    x = act1.forward(x);
////    x = dense2.forward(x);
////    x = relu2.forward(x);
//    x = dense3.forward(x);
//    x = softmax.forward(x);
//    std::cout << x << std::endl;

    auto x = dense[0].forward(input);
    for (size_t i = 1; i < dense.size(); ++i) {
        x = act[i - 1]->forward(x);
        x = dense[i].forward(x);
    }
    x = softmax.forward(x);
    return x;
}

void MLPClassifier::backward (const MLPClassifier::Matrix& real_value) {
    this->loss = this->calculate_loss(real_value);

    auto x = softmax.backward(real_value, settings.learning_rate);
    for (size_t i = dense.size() - 1; i > 0; --i) {
        x = dense[i].backward(x, settings.learning_rate);
        x = act[i - 1]->backward(x, settings.learning_rate);
    }
    dense[0].backward(x, settings.learning_rate);

//    auto x = softmax.backward(real_value, settings.learning_rate);
//    x = dense3.backward(x, settings.learning_rate);
////    x = relu2.backward(x, settings.learning_rate);
////    x = dense2.backward(x, settings.learning_rate);
//    x = act1.backward(x, settings.learning_rate);
//    dense1.backward(x, settings.learning_rate);
}

void MLPClassifier::on_epoch_callback (size_t epoch) {
    if (on_epoch_callback_) {
        on_epoch_callback_(this, epoch);
    }
}

double MLPClassifier::calculate_loss (const MLPClassifier::Matrix& real_value) const {
    return softmax.loss(real_value);
}


NetworkMetrics MLPClassifier::metrics (const Matrix& X_test, const Matrix& y_test)  {
    using namespace mlp::math;

    auto train_loss = this->loss;
    auto y_proba = this->forward(X_test);
    auto test_loss = this->calculate_loss(y_test);

    auto y_pred = argmax(y_proba, 0).as(f32);
    auto n = size(y_pred, 1);
    auto y_pred_per_class = zeros(10, n);
    gfor(seq i, 10) y_pred_per_class(i, span) = (y_pred == i).as(f32);

    auto tp = sum(y_pred_per_class * y_test, 1).as(f32);
    auto tn = sum((1 - y_pred_per_class) * (1 - y_test), 1).as(f32);
    auto fp = sum(y_pred_per_class * (1 - y_test), 1).as(f32);
    auto fn = sum((1 - y_pred_per_class) * y_test, 1).as(f32);

    auto precision_m = tp / (tp + fp), recall_m = tp / (tp + fn);

    auto y_labels_test = argmax(y_test, 0).as(f32);
    auto accuracy = sum<double>(y_labels_test == y_pred) / (double) n;
    auto precision = mean<double>(precision_m);

    auto recall = mean<double>(recall_m);
    auto f1_score = 2 * precision * recall / (precision + recall);

    return NetworkMetrics{train_loss, test_loss, accuracy, precision, recall, f1_score};
}

const char* MLPClassifier::act_name () const { return act[0]->name(); }
