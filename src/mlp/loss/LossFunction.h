//
// Created by Esteban on 6/5/2022.
//

#ifndef UNTITLED21_LOSSFUNCTION_H
#define UNTITLED21_LOSSFUNCTION_H

#include <boost/numeric/ublas/matrix.hpp>

class LossFunction {
public:
    using Matrix = boost::numeric::ublas::matrix<double>;
    using LossFunc = std::function<double (const Matrix&, const Matrix&)>;
    using LossGradient = std::function<Matrix (const Matrix&, const Matrix&)>;

    LossFunction(LossFunc loss_function, LossGradient derivative_function);

    [[nodiscard]]
    double loss(const Matrix& output, const Matrix& target) const;

    [[nodiscard]]
    Matrix derivative(const Matrix& output, const Matrix& target) const;
protected:
    LossFunc loss_function;
    LossGradient derivative_function;
};




#endif //UNTITLED21_LOSSFUNCTION_H
