// Copyright @SamsRepo. All rights reserved.

#include "NeuralNetwork.h"

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double y) {
    return y * (1.0 - y);
}

NeuralNetwork::NeuralNetwork(int inputSize, int hiddenSize, int outputSize)
    : inputSize(inputSize),
      hiddenSize(hiddenSize),
      outputSize(outputSize),
      weightsInputHidden(inputSize, std::vector<double>(hiddenSize)),
      weightsHiddenOutput(hiddenSize, std::vector<double>(outputSize)),
      hiddenLayer(hiddenSize),
      outputLayer(outputSize)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.05, 0.05);

    for (int i = 0; i < inputSize; ++i)
        for (int j = 0; j < hiddenSize; ++j)
            weightsInputHidden[i][j] = dis(gen);

    for (int i = 0; i < hiddenSize; ++i)
        for (int j = 0; j < outputSize; ++j)
            weightsHiddenOutput[i][j] = dis(gen);
}

void NeuralNetwork::saveWeights(const std::string& filename, int epoch) {
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) {
        file.write((char*)&epoch, sizeof(int));

        for (int i = 0; i < inputSize; ++i)
            for (int j = 0; j < hiddenSize; ++j)
                file.write((char*)&weightsInputHidden[i][j], sizeof(double));

        for (int i = 0; i < hiddenSize; ++i)
            for (int j = 0; j < outputSize; ++j)
                file.write((char*)&weightsHiddenOutput[i][j], sizeof(double));

        file.close();
    } else {
        std::cout << "Unable to open file for saving weights." << std::endl;
    }
}

// Load weights from a file
bool NeuralNetwork::loadWeights(const std::string& filename, int& loadedEpoch) {
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
        // Load epoch number
        file.read((char*)&loadedEpoch, sizeof(int));

        // Load weightsInputHidden
        for (int i = 0; i < inputSize; ++i)
            for (int j = 0; j < hiddenSize; ++j)
                file.read((char*)&weightsInputHidden[i][j], sizeof(double));

        // Load weightsHiddenOutput
        for (int i = 0; i < hiddenSize; ++i)
            for (int j = 0; j < outputSize; ++j)
                file.read((char*)&weightsHiddenOutput[i][j], sizeof(double));

        file.close();
        return true;
    } else {
        std::cout << "No saved weights found. Starting fresh training." << std::endl;
        return false;
    }
}

// Forward propagation
void NeuralNetwork::forward(std::vector<double>& input) {
    // Hidden layer
    for (int i = 0; i < hiddenSize; ++i) {
        hiddenLayer[i] = 0.0;
        for (int j = 0; j < inputSize; ++j) {
            hiddenLayer[i] += input[j] * weightsInputHidden[j][i];
        }
        hiddenLayer[i] = sigmoid(hiddenLayer[i]);
    }

    // Output layer
    for (int i = 0; i < outputSize; ++i) {
        outputLayer[i] = 0.0;
        for (int j = 0; j < hiddenSize; ++j) {
            outputLayer[i] += hiddenLayer[j] * weightsHiddenOutput[j][i];
        }
        outputLayer[i] = sigmoid(outputLayer[i]);
    }
}

// Backward propagation
void NeuralNetwork::backward(std::vector<double>& input, std::vector<double>& target, double learningRate) {
    // Calculate output errors
    std::vector<double> outputErrors(outputSize);
    for (int i = 0; i < outputSize; ++i) {
        outputErrors[i] = (target[i] - outputLayer[i]) * sigmoid_derivative(outputLayer[i]);
    }

    // Calculate hidden layer errors
    std::vector<double> hiddenErrors(hiddenSize);
    for (int i = 0; i < hiddenSize; ++i) {
        hiddenErrors[i] = 0.0;
        for (int j = 0; j < outputSize; ++j) {
            hiddenErrors[i] += outputErrors[j] * weightsHiddenOutput[i][j];
        }
        hiddenErrors[i] *= sigmoid_derivative(hiddenLayer[i]);
    }

    // Update weights between hidden and output layers
    for (int i = 0; i < hiddenSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            weightsHiddenOutput[i][j] += learningRate * outputErrors[j] * hiddenLayer[i];
        }
    }

    // Update weights between input and hidden layers
    for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < hiddenSize; ++j) {
            weightsInputHidden[i][j] += learningRate * hiddenErrors[j] * input[i];
        }
    }
}

// Training function
void NeuralNetwork::train(std::vector<std::vector<double>>& trainingInputs,
                          std::vector<std::vector<double>>& trainingOutputs,
                          int epochs, double learningRate) {
    int startEpoch = 0;
    int loadedEpoch = 0;
    if (loadWeights("saved_weights.bin", loadedEpoch)) {
        startEpoch = loadedEpoch;
        std::cout << "Resuming training from epoch " << startEpoch + 1 << std::endl;
    }

    for (int epoch = startEpoch; epoch < epochs; ++epoch) {
        double totalError = 0.0;
        for (size_t i = 0; i < trainingInputs.size(); ++i) {
            forward(trainingInputs[i]);
            backward(trainingInputs[i], trainingOutputs[i], learningRate);

            // Calculate total error (mean squared error)
            for (int j = 0; j < outputSize; ++j) {
                double error = trainingOutputs[i][j] - outputLayer[j];
                totalError += error * error;
            }
        }
        totalError /= trainingInputs.size();
        std::cout << "Epoch " << epoch + 1 << " Error: " << totalError << std::endl;

        // Save weights after each epoch
        saveWeights("saved_weights.bin", epoch + 1);
    }
}

// Prediction function
int NeuralNetwork::predict(std::vector<double>& input) {
    forward(input);
    // Find the index of the maximum output value
    return std::distance(outputLayer.begin(), std::max_element(outputLayer.begin(), outputLayer.end()));
}