//
// Created by Esteban on 6/7/2022.
//

#ifndef UNTITLED21_LOSSLAYER_H
#define UNTITLED21_LOSSLAYER_H

#include <mlp/net/Layer.h>

class LossLayer : public Layer {
public:
    using Matrix = Layer::Matrix;

public:
    [[nodiscard]]
    virtual double loss (const Matrix& real_value_) const = 0;
};

#endif //UNTITLED21_LOSSLAYER_H
