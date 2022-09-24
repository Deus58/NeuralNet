#pragma once

#include <vector>
#include <iostream>
#include <algorithm>
#include <string>
#include <cmath>
#include <cassert>

#include "../include/MathFunctions.h"



class Layer
{
public:
	Layer(int size, int prevSize, std::string activation);
	void backProp(const std::vector<double> targetValues);
	void feedForward(const std::vector<double> inputs);
	std::vector<double> getOutput() const;

private:
	std::vector<std::vector<double>> m_weights; // weight matrix
	std::vector<double> m_biases; // biases vector
	std::string m_activation; // activation function
	std::vector<double> m_outputs;
	std::vector<double> m_inputs;
};

