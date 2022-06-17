//
// Created by Esteban on 6/16/2022.
//
#include "input.h"
#include <vector>

static std::vector<std::string> parse_line (const std::string& line) {
    std::stringstream ss(line);
    std::vector<std::string> entries;

    for (std::string word; std::getline(ss, word, ',');) {
        entries.push_back(word);
    }
    return entries;
}

Dataset load_dataset (const char* path) {
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


mlp::math::Matrix to_categorical (const mlp::math::Matrix& y, int num_classes) {
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


