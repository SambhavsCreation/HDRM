// Copyright @SamsRepo. All rights reserved.

#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <random>


class NeuralNetwork {
public:
    NeuralNetwork(int inputSize, int hiddenSize, int outputSize);

    void train(std::vector<std::vector<double>>& trainingInputs,
               std::vector<std::vector<double>>& trainingOutputs,
               int epochs, double learningRate);

    int predict(std::vector<double>& input);

    void saveWeights(const std::string& filename, int epoch);
    bool loadWeights(const std::string& filename, int& loadedEpoch);

private:
    int inputSize;
    int hiddenSize;
    int outputSize;

    std::vector<std::vector<double>> weightsInputHidden;
    std::vector<std::vector<double>> weightsHiddenOutput;

    std::vector<double> hiddenLayer;
    std::vector<double> outputLayer;

    void forward(std::vector<double>& input);
    void backward(std::vector<double>& input, std::vector<double>& target, double learningRate);
};



#endif //NEURALNETWORK_H
