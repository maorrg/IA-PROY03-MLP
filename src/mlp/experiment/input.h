//
// Created by Esteban on 6/16/2022.
//

#ifndef UNTITLED21_INPUT_H
#define UNTITLED21_INPUT_H

#include <fstream>
#include <mlp/math/mlp_math.h>

std::vector<std::string>
parse_line (const std::string& line) {
    std::stringstream ss(line);
    std::vector<std::string> entries;

    for (std::string word; std::getline(ss, word, ',');) {
        entries.push_back(word);
    }
    return entries;
}

std::pair<mlp::math::Matrix, mlp::math::Matrix>
load_dataset (const char* path) {
    std::vector<std::vector<double>> X_;
    std::vector<int> Y_;

    std::fstream file(path, std::ios::in);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file");
    }

    for (std::string line; getline(file, line);) {
        auto entries = parse_line(line);

        auto yi = std::stoi(entries[entries.size() - 1]);
        Y_.push_back(yi);
        entries.pop_back();

        std::vector<double> xi;
        xi.reserve(entries.size());
        for (const auto& entry : entries) {
            xi.push_back(std::stod(entry));
        }
        X_.push_back(std::move(xi));
    }
    mlp::math::Matrix X(X_[0].size(), X_.size());
    mlp::math::Matrix Y = mlp::math::zeros(10, Y_.size());

    for (size_t i = 0; i < X_.size(); i++) {
        for (size_t j = 0; j < X_[0].size(); j++) {
            X(j, i) = X_[i][j];
        }
    }
    for (size_t i = 0; i < Y_.size(); i++) {
        Y(Y_[i], i) = 1;
    }
    return std::make_pair(X, Y);
}


#endif //UNTITLED21_INPUT_H
