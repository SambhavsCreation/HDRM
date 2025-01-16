cmake_minimum_required(VERSION 3.29)
project(HDRM)

set(CMAKE_CXX_STANDARD 20)
#find_library(OpenCV)
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(HDRM main.cpp
        NeuralNetwork.cpp
        NeuralNetwork.h
        MNISTReaders.cpp
        MNISTReader.h)

configure_file(train-images-idx3-ubyte train-images-idx3-ubyte COPYONLY)
configure_file(train-labels-idx1-ubyte train-labels-idx1-ubyte COPYONLY)
configure_file(t10k-labels-idx1-ubyte t10k-labels-idx1-ubyte COPYONLY)
configure_file(t10k-images-idx3-ubyte t10k-images-idx3-ubyte COPYONLY)
configure_file(0.png 0.png COPYONLY)

target_link_libraries(HDRM ${OpenCV_LIBS})

