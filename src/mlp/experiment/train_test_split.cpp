#include "train_test_split.h"


DatasetSplit train_test_split (const Dataset& dataset, double train_ratio) {
    auto shuffled = shuffle_dataset(dataset);
    auto n = mlp::math::size(shuffled.data, 0);
    auto train_size = (int) ((float) n * train_ratio);
#if defined (MLP_USE_BOOST_BACKEND)
    auto test_size = n - train_size;
    Dataset train = {
        mlp::math::Matrix(train_size, mlp::math::size(shuffled.data, 1)),
        mlp::math::Matrix(train_size, mlp::math::size(shuffled.labels, 1))
    };
    for (int i = 0; i < (int) train_size; i++) {
        mlp::math::row(train.data, i) = mlp::math::row(shuffled.data, i);
        mlp::math::row(train.labels, i) = mlp::math::row(shuffled.labels, i);
    }
    Dataset test = {
        mlp::math::Matrix(test_size, mlp::math::size(shuffled.data, 1)),
        mlp::math::Matrix(test_size, mlp::math::size(shuffled.labels, 1))
    };
    for (int i = 0; i < (int) test_size; i++) {
        mlp::math::row(test.data, i) = mlp::math::row(shuffled.data, i + train_size);
        mlp::math::row(test.labels, i) = mlp::math::row(shuffled.labels, i + train_size);
    }
#else
    Dataset train = { shuffled.data(mlp::math::seq(0, (double)train_size-1), mlp::math::span),
                      shuffled.labels(mlp::math::seq(0, (double)train_size-1), mlp::math::span) };
    Dataset test = { shuffled.data(mlp::math::seq(train_size, (double)n-1), mlp::math::span),
                     shuffled.labels(mlp::math::seq(train_size, (double)n-1), mlp::math::span) };
#endif
    return {train, test};
}

Dataset shuffle_dataset (const Dataset& dataset) {
    auto indices = mlp::math::shuffle_idx((int) mlp::math::size(dataset.data, 0));
#if defined (MLP_USE_BOOST_BACKEND)
    Dataset shuffled = {
        mlp::math::Matrix(mlp::math::size(dataset.data, 0), mlp::math::size(dataset.data, 1)),
        mlp::math::Matrix(mlp::math::size(dataset.labels, 0), mlp::math::size(dataset.labels, 1))
    };

    for (int i = 0; i < (int) mlp::math::size(dataset.data, 0); i++) {
        mlp::math::row(shuffled.data, i) = mlp::math::row(dataset.data, indices[i]);
        mlp::math::row(shuffled.labels, i) = mlp::math::row(dataset.labels, indices[i]);
    }
#else
    Dataset shuffled = {dataset.data(indices, mlp::math::span), dataset.labels(indices, mlp::math::span)};
#endif
    return shuffled;
}