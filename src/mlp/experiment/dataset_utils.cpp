//
// Created by Esteban on 6/16/2022.
//
#include "dataset_utils.h"
#include <vector>
#include <filesystem>

static std::vector<std::string> parse_line (const std::string& line) {
    std::stringstream ss(line);
    std::vector<std::string> entries;

    for (std::string word; std::getline(ss, word, ',');) {
        entries.push_back(word);
    }
    return entries;
}

Dataset load_dataset_from_file (const char* path) {
    std::vector<std::vector<double>> df;

    std::fstream file(path, std::ios::in);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file");
    }

    for (std::string line; getline(file, line);) {
        auto entries = parse_line(line);

        std::vector<double> record;
        record.reserve(entries.size());
        for (const auto& entry : entries) {
            record.push_back(std::stod(entry));
        }
        df.emplace_back(std::move(record));
    }

    Dataset dataset {
        mlp::math::Matrix(df.size(), df[0].size()),
        mlp::math::Matrix(df.size(), 1)
    };

    for (int i = 0; i < (int) df.size(); i++) {
        for (int j = 0; j < (int) df[0].size() - 1; j++) {
            dataset.data(i, j) = df[i][j];
        }
        dataset.labels(i, 0) = df[i][df[0].size() - 1];
    }
    return dataset;
}


Dataset::Matrix to_categorical (const Dataset::Matrix& y, int num_classes) {
    auto y_cat = mlp::math::zeros(mlp::math::size(y, 0), num_classes);
#if defined (MLP_USE_BOOST_BACKEND)
    for (size_t i = 0; i < mlp::math::size(y, 0); i++) {
        y_cat(i, (int) y(i, 0)) = 1;
    }
#else
    gfor(mlp::math::seq i, num_classes) {
        af::array condition = y == i;
        y_cat(mlp::math::span, i) = condition.as(f32);
    }
#endif
    return y_cat;
}

Dataset load_dataset (const char* path, const char* cache, bool use_cache) {
    std::cout << "Loading dataset from " << path << " (Cache: " << cache << ")" << std::endl;
    if (!std::filesystem::exists(cache) || !use_cache) {
        std::cout << "Cache for dataset was " << (use_cache ? "not found"
                                                            : "ignored") << ". Loading from file..." << std::endl;
        auto Xy = load_dataset_from_file(path);
        std::cout << "Saving cache..." << std::endl;
        mlp::math::saveArray("x", Xy.data, cache, false);
        mlp::math::saveArray("y", Xy.labels, cache, true);
        std::cout << "Cache saved." << std::endl;
        Xy.labels = to_categorical(Xy.labels, 10);
        return Xy;
    }
    else {
        auto ds = Dataset{
            mlp::math::readArray("../Xy.cache", "x"),
            mlp::math::readArray("../Xy.cache", "y")
        };
        ds.labels = to_categorical(ds.labels, 10);
        return ds;
    }
}


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

