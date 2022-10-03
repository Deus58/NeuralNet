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
TODO:
* Matrix class
* Vector class
* back propagation using gradient descent
*/

/*
Notes:
* partial derivative of a sum is the sum of derivatives
d/dx sum(x, y) = sum(d/dx * x, d/dx * y) = 1 + 0

* in partial derivative of multiplication the other variables are constants
d/dx MUL(x, y) = d/dx (x * y) = y * d/dx * x = y * 1 = y

* Partial derivative of max is 1 if the variable we derivate by is the larger else 0
d/dx * max(x, y) = 1 * (x > y)

* The gradient is the vector of all the possible partial derivatives 


each neuron is ACTIVATION_FUNCTION(sum from 0 to n (w_n * x_n) + b)

to get the gradiant is the partial derivative for each x, w, b

we need to calculate the partial derivative for each atomic operation to 
get their impact on the output

lets take for example a network of 3 3 1 with first layer input and the 
2 other layers are using ReLU 

we take the mean squared error from the output the derivative of the ReLU function in this layer

*/


int main()
{
	// Network net({4, 3, 1}, {"input","ReLU","ReLU"});
	std::vector<double> inp;
	inp.push_back(2);
	inp.push_back(4);
	inp.push_back(6);
	inp.push_back(8);

	std::vector<double> expected; // f(x,y,z) = 3x + 4y + 2z + 12
	expected.push_back(46);
	// expected.push_back(24);
	// expected.push_back(30);

	// net.feedForward(inp);

	// std::vector<double> res = net.getOutput();
	
	// printDoubleVec(res);

	Layer L1(4, 0, "input");
	Layer L2(3, 4, "ReLU");

	L1.feedForward(inp);
	L2.feedForward(L1.getOutput());

	printDoubleVec(L2.getOutput());
	printDoubleMat(L2.getWeights());

	
	printDoubleMat(transposeMatrix(L2.getWeights()));


	L2.backProp(std::vector<double>({1.f, 1.f, 1.f}));

	return 0;

}



/*
Sources :
Neural Networks from Scartch in Python - Harrison Kinsley & Daniel Kukiela

*/