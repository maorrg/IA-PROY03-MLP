cmake_minimum_required(VERSION 3.20)
project(untitled21)

set(CMAKE_CXX_STANDARD 17)
find_package(Boost REQUIRED)

add_executable(untitled21
        main.cpp
        mlp/Network.cpp
        mlp/activation/ActivationLayer.cpp
        mlp/activation/Tanh.cpp
        mlp/activation/SigmoidLayer.cpp
        mlp/activation/ReLU.cpp
        mlp/loss/MSELoss.cpp
        mlp/loss/SoftmaxLoss.cpp
        mlp/net/DenseLayer.cpp mlp/math/Matrix.cpp mlp/math/Matrix.h mlp/math/math.h)

target_include_directories(untitled21 PUBLIC
        ${Boost_INCLUDE_DIRS})

#target_compile_options(untitled21 PRIVATE
#        $<$<CXX_COMPILER_ID:MSVC>: /W4 /WX>
#        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>: -Wall -Wextra -Wpedantic -Werror>)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CXX_COMPILER_ID}")