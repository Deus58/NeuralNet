#define _USE_MATH_DEFINES

#include <vector>
#include <iostream>
#include <algorithm>
#include <string>
#include <cmath>
#include <cassert>
#include "../include/Layer.h"
#include "../include/Network.h"

#define assertm(exp, msg) assert(((void)msg, exp))

/* 
Notes:
Dense neural network
input layer -> sort of empty neurons
hidden layers -> for each neuron activation(sum(each value from previous layer * weight) + bias)
output -> just like hidden except they represent the end result of the network
each neuron is fully connected to the previous layer
weights are initialized to 1 and bias to 0
forward feed -> move data through network
back propagation -> teach the network using expected values vs result
an entire layer can be summed and moved to the next layer to each neuron
this can also be represented by Vec_Activation([[Weights]] * [inputs] + [biases])
*/



int main()
{
	Network net({1, 3, 1}, {"input","ReLU","ReLU"});
	std::vector<double> inp;
	inp.push_back(2);
	inp.push_back(4);
	inp.push_back(6);

	std::vector<double> expected; // y = 3x + 12
	expected.push_back(18);
	expected.push_back(24);
	expected.push_back(30);

	net.feedForward(inp);

	std::vector<double> res = net.getOutput();

	return 0;

}