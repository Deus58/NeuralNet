#include "../include/Network.h"


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