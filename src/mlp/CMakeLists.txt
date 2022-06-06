cmake_minimum_required(VERSION 3.20)
project(untitled21)

set(CMAKE_CXX_STANDARD 20)

add_executable(untitled21 main.cpp layer/Layer.cpp layer/Layer.h layer/activation/ActivationLayer.cpp layer/activation/ActivationLayer.h layer/DenseLayer.cpp layer/DenseLayer.h Network.cpp Network.h layer/activation/TanhLayer.h layer/activation/SoftmaxLayer.cpp layer/activation/SoftmaxLayer.h utils/Utilities.h layer/activation/TanhLayer.cpp loss/LossFunction.cpp loss/LossFunction.h layer/activation/SigmoidLayer.cpp layer/activation/SigmoidLayer.h layer/activation/ReluLayer.cpp layer/activation/ReluLayer.h utils/Utilities.cpp loss/MSE.cpp loss/MSE.h loss/CrossEntropy.cpp loss/CrossEntropy.h)

target_compile_options(untitled21 PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>: /W4 /WX>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>: -Wall -Wextra -Wpedantic -Werror>)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CXX_COMPILER_ID}")