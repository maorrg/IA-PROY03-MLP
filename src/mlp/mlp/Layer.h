//
// Created by Esteban on 6/4/2022.
//

#ifndef UNTITLED21_LAYER_H
#define UNTITLED21_LAYER_H

#include <boost/numeric/ublas/matrix.hpp>

class Layer {
public:
    using Matrix = boost::numeric::ublas::matrix<double>;

public:
    Layer () = default;
    virtual ~Layer () = default;

    virtual Matrix forward (const Matrix& input) = 0;
    virtual Matrix backward (const Matrix& gradient, double learning_rate) = 0;
};


#endif //UNTITLED21_LAYER_H
