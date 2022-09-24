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


int main()
{
	Network net({3, 3, 1}, {"input","ReLU","ReLU"});
	std::vector<double> inp;
	inp.push_back(2);
	inp.push_back(4);
	inp.push_back(6);

	std::vector<double> expected; // f(x,y,z) = 3x + 4y + 2z + 12
	expected.push_back(18);
	expected.push_back(24);
	expected.push_back(30);

	net.feedForward(inp);

	std::vector<double> res = net.getOutput();
	
	printDoubleVec(res);

	return 0;

}