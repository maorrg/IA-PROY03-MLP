//
// Created by Esteban on 6/15/2022.
//

#ifndef UNTITLED21_MLP_MATH_H
#define UNTITLED21_MLP_MATH_H

#if defined(MLP_USE_BOOST_BACKEND)
#include <mlp/math/boost_backend.h>
#elif defined(MLP_USE_ARRAYFIRE_BACKEND)
#include <mlp/math/arrayfire_backend.h>
#endif
#endif