#ifndef UNTITLED21_SPLIT_H
#define UNTITLED21_SPLIT_H

#include "data.h"

Dataset shuffle_dataset (const Dataset& dataset);

DatasetSplit split_dataset (const Dataset& dataset, double train_ratio);

#endif //UNTITLED21_SPLIT_H
