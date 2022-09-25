#include "../include/Layer.h"


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

void Layer::backProp(const std::vector<double> targetValues)
{
	double mse = meanSquaredErrorVec(m_outputs, targetValues);
}