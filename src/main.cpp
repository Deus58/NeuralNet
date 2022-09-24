#define _USE_MATH_DEFINES

#include <vector>
#include <iostream>
#include <algorithm>
#include <string>
#include <cmath>
#include <cassert>

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

void printDoubleVec(const std::vector<double> &vec)
{
	for (double d : vec) std::cout << d << "\n";
	std::cout << std::endl;
}

std::vector<double> doubleVectorAddition(const std::vector<double> &vec1, const std::vector<double> &vec2)
{
	assertm(vec1.size() == vec2.size(), "[USER ERROR]: Vector one and Vector two are not of the same size");
	std::vector<double> result;
	for(int i = 0; i < vec1.size(); i++) result.push_back(vec1[i] + vec2[i]);

}

std::vector<double> doubleVectorTimesMatrix(const std::vector<double> &vec, const std::vector<std::vector<double>> &mat)
{
	std::vector<double> result;
	int rowLength = mat[0].size();

	if(rowLength != vec.size()) return std::vector<double>();

	double sum = 0;
	for(int i = 0; i < mat.size(); i++)
	{
		sum = 0;
		for(int j = 0; j < rowLength; j++) sum = sum + (vec[j] * mat[i][j]);
		result.push_back(sum);
	}
	return result;
}

double RandomDouble(double a, double b) {
    double random = ((double) rand()) / (double) RAND_MAX;
    double diff = b - a;
    double r = random * diff;
    return a + r;
}


std::vector<double> ReLU(std::vector<double> inputs) // ReLu on vector
{
	std::vector<double> result;
	for (double d : inputs)
	{
		if(d < 0) result.push_back(0);
		else result.push_back(d);
	}

	return result;
} 

std::vector<double> DReLU(std::vector<double> inputs) // ReLu derivative on vector
{
	std::vector<double> result;
	for (double d : inputs)
	{
		if(d < 0) result.push_back(0);
		else result.push_back(1);
	}

	return result;
}

double meanSquaredErrorVec(std::vector<double> output, std::vector<double> target)
{
	if(output.size() != target.size()) return -1.f;

	double sum = 0;

	for(int i = 0; i < output.size(); i++)
	{
		sum = sum + std::pow((target[i] - output[i]), 2);
	}
	return sum / output.size();

}


std::vector<double> Softmax(std::vector<double> inputs) // Softmax on vector
{
	double sum = 0;
	for(double d : inputs) sum = sum + d;

	std::vector<double> result;
	
	double mx = *std::max(inputs.begin(), inputs.end());

	for(double d : inputs) result.push_back(std::pow(M_E, d - mx));
	return result;
}

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

Layer::Layer(int size, int prevSize, std::string activation)
: m_activation(activation)
{
	if(activation != "input") // initialize weights and biases
	{
		for(int i = 0; i < size; i++)
		{
			this->m_biases.push_back(RandomDouble(0.01f, 1.f));
			this->m_weights.push_back(std::vector<double>());

			for(int j = 0; j < prevSize; j++)
			{
				this->m_weights[i].push_back(RandomDouble(0.01f, 1.f));
			}
		}
	}
}

void Layer::feedForward(std::vector<double> inputs)
{
	this->m_inputs = inputs;
	if(this->m_activation == "input")
	{
		this->m_outputs = inputs;
		return;
	}

	if(this->m_activation == "ReLU")
	{
		this->m_outputs = ReLU(doubleVectorAddition(doubleVectorTimesMatrix(inputs, this->m_weights), this->m_biases));
		return;
	}

	if(this->m_activation == "Softmax")
	{
		this->m_outputs = Softmax(doubleVectorAddition(doubleVectorTimesMatrix(inputs, this->m_weights), this->m_biases));
		return;
	}
}

std::vector<double> Layer::getOutput() const
{
	return this->m_outputs; 
}


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

Network::Network(std::vector<int> structure, std::vector<std::string> activations)
{
	assertm(activations[0] == "input", "[USER ERROR]: First Layer Is Not Input");

	this->m_layers.push_back(Layer(structure[0], 0, activations[0])); // input layer

	for (int i = 1; i < structure.size(); i++)
	{
		this->m_layers.push_back(Layer(structure[i], structure[i-1], activations[i])); // other layers

	}
}

void Network::feedForward(std::vector<double> inputs)
{
	this->m_outputs = inputs;
	for(int i = 0; i < this->m_layers.size(); i++)
	{
		this->m_layers[i].feedForward(this->m_outputs);
		this->m_outputs = this->m_layers[i].getOutput();
	}
}

std::vector<double> Network::getOutput() const
{
	return this->m_outputs; 
}

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

	printDoubleVec(res);

	return 0;

}