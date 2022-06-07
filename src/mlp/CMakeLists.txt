cmake_minimum_required(VERSION 3.20)
project(untitled21)

set(CMAKE_CXX_STANDARD 20)

add_executable(untitled21
        main.cpp
        mlp/Network.cpp
        mlp/activation/ActivationLayer.cpp
        mlp/activation/SoftmaxLayer.cpp
        mlp/activation/TanhLayer.cpp
        mlp/activation/SigmoidLayer.cpp
        mlp/activation/ReluLayer.cpp
        mlp/loss/MSELoss.cpp
        mlp/loss/SoftmaxLoss.cpp
        mlp/net/DenseLayer.cpp)

target_compile_options(untitled21 PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>: /W4 /WX>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>: -Wall -Wextra -Wpedantic -Werror>)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CXX_COMPILER_ID}")