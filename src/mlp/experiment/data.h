#ifndef UNTITLED21_DATA_H
#define UNTITLED21_DATA_H

#include <mlp/math/mlp_math.h>

struct Dataset {
    mlp::math::Matrix data;
    mlp::math::Matrix labels;
};

struct DatasetSplit {
    Dataset train;
    Dataset test;
};

#endif //UNTITLED21_DATA_H
