//
// Created by Esteban on 6/16/2022.
//

#ifndef UNTITLED21_DATASET_UTILS_H
#define UNTITLED21_DATASET_UTILS_H

#include <fstream>
#include <mlp/math/mlp_math.h>

struct Dataset {
    using Matrix = mlp::math::Matrix;
    mlp::math::Matrix data;
    mlp::math::Matrix labels;
};

struct DatasetSplit {
    Dataset train;
    Dataset test;
};

Dataset load_dataset (const char* path, const char* cache, bool use_cache);

Dataset load_dataset_from_file (const char* path);

Dataset::Matrix to_categorical (const Dataset::Matrix& mat, int num_classes);

Dataset shuffle_dataset (const Dataset& dataset);

DatasetSplit train_test_split (const Dataset& dataset, double train_ratio);

#endif //UNTITLED21_DATASET_UTILS_H
