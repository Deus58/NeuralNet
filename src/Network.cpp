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


// This is code copied from another project, need to chagnge


// double Neuron::eta = 0.15;  
// double Neuron::alpha = 0.5;   


// void Neuron::updateInputWeights(Layer &prevLayer)
// {

//     for (unsigned n = 0; n < prevLayer.size(); ++n) {
//         Neuron &neuron = prevLayer[n];
//         double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

//         double newDeltaWeight =

//         neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
//         neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
//     }
// }

// double Neuron::sumDOW(const Layer &nextLayer) const
// {
//     double sum = 0.0;


//     for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
//         sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
//     }

//     return sum;
// }

// void Neuron::calcHiddenGradients(const Layer &nextLayer)
// {
//     double dow = sumDOW(nextLayer);
//     m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
// }

// void Neuron::calcOutputGradients(double targetVal)
// {
//     double delta = targetVal - m_outputVal;
//     m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
// }

// double Neuron::transferFunction(double x)
// {

//     return tanh(x);
// }

// double Neuron::transferFunctionDerivative(double x)
// {
//     return 1.0 - x * x;
// }

// void Net::backProp(const vector<double> &targetVals)
// {

//     Layer &outputLayer = m_layers.back();
//     m_error = 0.0;

//     for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
//         double delta = targetVals[n] - outputLayer[n].getOutputVal();
//         m_error += delta * delta;
//     }
//     m_error /= outputLayer.size() - 1; // get average error squared
//     m_error = sqrt(m_error); // RMS

//     m_recentAverageError =
//             (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
//             / (m_recentAverageSmoothingFactor + 1.0);

//     for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
//         outputLayer[n].calcOutputGradients(targetVals[n]);
//     }

//     for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
//         Layer &hiddenLayer = m_layers[layerNum];
//         Layer &nextLayer = m_layers[layerNum + 1];

//         for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
//             hiddenLayer[n].calcHiddenGradients(nextLayer);
//         }
//     }

//     for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
//         Layer &layer = m_layers[layerNum];
//         Layer &prevLayer = m_layers[layerNum - 1];

//         for (unsigned n = 0; n < layer.size() - 1; ++n) {
//             layer[n].updateInputWeights(prevLayer);
//         }
//     }
// }
