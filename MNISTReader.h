//
// Created by Sambhav Sharma on 14/01/25.
//

#ifndef MNISTREADER_H
#define MNISTREADER_H
#include <string>

std::vector<double> preprocessImage(const std::string& imagePath);
void readMNISTLabels(const std::string& filename, std::vector<std::vector<double>>& labels, int numClasses);
void readMNISTImages(const std::string& filename, std::vector<std::vector<double>>& images);
int reverseInt(int i);

#endif //MNISTREADER_H
