#include <iostream>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>

#include "Network.h"
#include "layer/DenseLayer.h"
#include "layer/activation/SoftmaxLayer.h"
#include "layer/activation/TanhLayer.h"
#include "layer/activation/ReluLayer.h"
#include "layer/activation/SigmoidLayer.h"
#include "loss/MSE.h"
#include "loss/CrossEntropy.h"

int main () {
    using namespace boost::numeric::ublas;
//    if (!freopen("../out.txt", "w", stdout)) std::abort();

    DenseLayer L1(2, 4);
    TanhLayer L2;
    DenseLayer L3(4, 4);
    TanhLayer L4;
    DenseLayer L5(4, 2);
    SigmoidLayer L6;
    LossFunction loss_function = MSE();

    Network network({ &L1, &L2, &L3, &L4, &L5, &L6 }, loss_function);
    DenseLayer::Matrix input{2, 4};
    DenseLayer::Matrix output{2, 4};
    input(0, 0) = 1; input(1, 0) = 1; output(0, 0) = 1; output(1, 0) = 0;
    input(0, 1) = 0; input(1, 1) = 1; output(0, 1) = 0; output(1, 1) = 1;
    input(0, 2) = 1; input(1, 2) = 0; output(0, 2) = 0; output(1, 2) = 1;
    input(0, 3) = 0; input(1, 3) = 0; output(0, 3) = 1; output(1, 3) = 0;

    network.train(input, output, 10000, 0.1);
    std::cout << network.predict(input) << std::endl;
}