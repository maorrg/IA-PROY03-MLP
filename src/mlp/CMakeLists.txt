cmake_minimum_required(VERSION 3.20)
project(untitled21)
set(CMAKE_CXX_STANDARD 17)

set(PROJECT_SOURCES
        mlp/math/arrayfire_backend.cpp
        mlp/math/boost_backend.cpp
        mlp/net/DenseLayer.cpp
        mlp/loss/SoftmaxLoss.cpp
        mlp/loss/MSELoss.cpp
        mlp/activation/ReLU.cpp
        mlp/activation/Sigmoid.cpp
        mlp/activation/Tanh.cpp
        mlp/net/Network.cpp
        experiment/NetworkReLU.cpp
        experiment/input.cpp
        experiment/train_test_split.cpp
        main.cpp
        )

add_executable(untitled21 ${PROJECT_SOURCES})
target_include_directories(untitled21 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(untitled21 PRIVATE $<$<CXX_COMPILER_ID:MSVC>: /W4> $<$<NOT:$<CXX_COMPILER_ID:MSVC>>: -Wall -Wextra -Wpedantic>)


find_package(ArrayFire)
if (NOT ArrayFire_FOUND OR MLP_USE_BOOST)
    add_definitions(-DMLP_USE_BOOST_BACKEND=1)
    find_package(Boost REQUIRED)
    target_include_directories(untitled21 PUBLIC ${Boost_INCLUDE_DIRS})
    target_link_libraries(untitled21 ${Boost_LIBRARIES})
    message(STATUS "Using Boost version: ${Boost_VERSION}")
else()
    add_definitions(-DMLP_USE_ARRAYFIRE_BACKEND=1)
    target_include_directories(untitled21 PUBLIC ${ArrayFire_INCLUDE_DIRS})
    target_link_libraries(untitled21 ${ArrayFire_LIBRARIES})
    message(STATUS "Using ArrayFire version: ${ArrayFire_VERSION}")

endif()


message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Current source dir: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "Boost include dir: ${Boost_INCLUDE_DIRS}")