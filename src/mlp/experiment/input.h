//
// Created by Esteban on 6/16/2022.
//

#ifndef UNTITLED21_INPUT_H
#define UNTITLED21_INPUT_H

#include <fstream>
#include "data.h"

Dataset load_dataset_from_file (const char* path);

mlp::math::Matrix to_categorical (const mlp::math::Matrix& mat, int num_classes);



#endif //UNTITLED21_INPUT_H
