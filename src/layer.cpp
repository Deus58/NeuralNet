#include "../include/Layer.h"


void printDoubleVec(const std::vector<double> &vec)
{
	for (double d : vec) std::cout << d << "\n";
	std::cout << std::endl;
}

std::vector<double> doubleVectorAddition(const std::vector<double> &vec1, const std::vector<double> &vec2)
{
	assertm(vec1.size() != vec2.size(), "[USER ERROR]: Vector one and Vector two are not of the same size");
	std::vector<double> result;
	for(int i = 0; i < vec1.size(); i++) result.push_back(vec1[i] + vec2[i]);
	return result;

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