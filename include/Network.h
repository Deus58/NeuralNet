#pragma once

#include <vector>
#include <iostream>
#include <algorithm>
#include <string>
#include <cmath>
#include <cassert>

#include "../include/Layer.h"





class Network
{
public:
	Network(std::vector<int> structure, std::vector<std::string> activations);
	void train(std::vector<std::vector<double>> inputs);
	std::vector<double> getOutput() const;
	void feedForward(std::vector<double> inputs);

private:	
	std::vector<Layer> m_layers;
	void backProp(std::vector<double> targetValues);
	std::vector<double> m_outputs;

};

