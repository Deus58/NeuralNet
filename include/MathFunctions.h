#pragma once

#include <vector>
#include <iostream>
#include <algorithm>
#include <string>
#include <cmath>
#include <cassert>

#define assertm(exp, msg) assert(((void)msg, exp))

void printDoubleVec(const std::vector<double> &vec);

std::vector<double> doubleVectorAddition(const std::vector<double> &vec1, const std::vector<double> &vec2);

std::vector<double> doubleVectorTimesMatrix(const std::vector<double> &vec, const std::vector<std::vector<double>> &mat);

double RandomDouble(double a, double b);

std::vector<double> ReLU(std::vector<double> inputs);

std::vector<double> DReLU(std::vector<double> inputs);

double meanSquaredErrorVec(std::vector<double> output, std::vector<double> target);

std::vector<double> Softmax(std::vector<double> inputs);