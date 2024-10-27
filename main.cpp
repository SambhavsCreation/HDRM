// Copyright @SamsRepo. All rights reserved.

#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include "NeuralNetwork.h"
#include "MNISTReaders.cpp"

int main() {
    std::vector<std::vector<double>> trainingImages;
    std::vector<std::vector<double>> trainingLabels;

    readMNISTImages("train-images-idx3-ubyte", trainingImages);
    readMNISTLabels("train-labels-idx1-ubyte", trainingLabels, 10);

    int inputSize = 28 * 28;
    int hiddenSize = 64;
    int outputSize = 10;

    NeuralNetwork nn(inputSize, hiddenSize, outputSize);

    int epochs = 5;
    double learningRate = 0.1;

    nn.train(trainingImages, trainingLabels, epochs, learningRate);

    std::vector<std::vector<double>> testImages;
    std::vector<std::vector<double>> testLabels;

    readMNISTImages("t10k-images-idx3-ubyte", testImages);
    readMNISTLabels("t10k-labels-idx1-ubyte", testLabels, 10);

    int correct = 0;

    for (size_t i = 0; i < testImages.size(); ++i) {
        int predicted = nn.predict(testImages[i]);

        int actual = std::distance(testLabels[i].begin(), std::max_element(testLabels[i].begin(), testLabels[i].end()));

        if (predicted == actual) {
            ++correct;
        }
    }

    double accuracy = (double)correct / testImages.size() * 100.0;
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;

    std::string imagePath;
    std::cout << "Enter the path to a PNG image of a handwritten digit: ";
    std::getline(std::cin, imagePath);

    std::vector<double> userInput = preprocessImage(imagePath);

    int predictedDigit = nn.predict(userInput);
    std::cout << "The predicted digit is: " << predictedDigit << std::endl;

    return 0;
}