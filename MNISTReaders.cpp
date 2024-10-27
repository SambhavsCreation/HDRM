// Copyright @SamsRepo. All rights reserved.

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>

int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void readMNISTImages(const std::string& filename, std::vector<std::vector<double>>& images) {
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
        int magicNumber = 0;
        int numberOfImages = 0;
        int nRows = 0;
        int nCols = 0;

        file.read((char*)&magicNumber, sizeof(magicNumber));
        magicNumber = reverseInt(magicNumber);

        file.read((char*)&numberOfImages, sizeof(numberOfImages));
        numberOfImages = reverseInt(numberOfImages);

        file.read((char*)&nRows, sizeof(nRows));
        nRows = reverseInt(nRows);

        file.read((char*)&nCols, sizeof(nCols));
        nCols = reverseInt(nCols);

        for (int i = 0; i < numberOfImages; ++i) {
            std::vector<double> image(nRows * nCols);
            for (int r = 0; r < nRows * nCols; ++r) {
                unsigned char temp = 0;
                file.read((char*)&temp, sizeof(temp));
                image[r] = (double)temp / 255.0; // Normalize pixel values
            }
            images.push_back(image);
        }
    } else {
        std::cout << "Unable to open file " << filename << std::endl;
        exit(1);
    }
}

void readMNISTLabels(const std::string& filename, std::vector<std::vector<double>>& labels, int numClasses) {
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
        int magicNumber = 0;
        int numberOfItems = 0;

        file.read((char*)&magicNumber, sizeof(magicNumber));
        magicNumber = reverseInt(magicNumber);

        file.read((char*)&numberOfItems, sizeof(numberOfItems));
        numberOfItems = reverseInt(numberOfItems);

        for (int i = 0; i < numberOfItems; ++i) {
            unsigned char temp = 0;
            file.read((char*)&temp, sizeof(temp));
            std::vector<double> label(numClasses, 0.0);
            label[(int)temp] = 1.0; // One-hot encoding
            labels.push_back(label);
        }
    } else {
        std::cout << "Unable to open file " << filename << std::endl;
        exit(1);
    }
}

std::vector<double> preprocessImage(const std::string& imagePath) {
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_GRAYSCALE); // Load image in grayscale
    if (img.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        exit(1);
    }

    // Resize image to 28x28
    cv::Mat resizedImg;
    cv::resize(img, resizedImg, cv::Size(28, 28));

    // Invert colors if necessary (MNIST digits are white on black background)
    // You might need to invert the image depending on your input image
    // Uncomment the following line to invert the image
    // cv::bitwise_not(resizedImg, resizedImg);

    // Normalize pixel values to [0, 1]
    resizedImg.convertTo(resizedImg, CV_64F, 1.0 / 255.0);

    // Flatten the image into a single vector
    std::vector<double> inputVector;
    inputVector.reserve(28 * 28);
    for (int i = 0; i < resizedImg.rows; ++i) {
        for (int j = 0; j < resizedImg.cols; ++j) {
            inputVector.push_back(resizedImg.at<double>(i, j));
        }
    }

    return inputVector;
}