#include "../include/MathFunctions.h"

void printDoubleVec(const std::vector<double> &vec)
{
	for (double d : vec) std::cout << d << "\n";
	std::cout << std::endl;
}

void printDoubleMat(const std::vector<std::vector<double>> &mat)
{
	for (const std::vector<double> &v : mat)
	{
		for (double d : v)
		{
			std::cout << d << "  ";

		}
		std::cout << "\n";

	} 
	std::cout << std::endl;
}

std::vector<std::vector<double>> transposeMatrix(const std::vector<std::vector<double>> &mat)
{

	if(mat.size() == mat[0].size())
	{
		std::vector<std::vector<double>> result(mat);

		for (int c = 0; c < mat.size(); c++)
		{
			for(int r = 0; r < mat[0].size(); r++)
			{
				result[c][r] = mat[r][c];
			}

		}


		return result;
	}
	std::vector<std::vector<double>> result;

	for(int r = 0; r < mat[0].size(); r++)
	{
		std::vector<double> newRow;
		for (int c = 0; c < mat.size(); c++)
		{
			newRow.push_back(mat[c][r]);
		}
		result.push_back(newRow);
	}
	return result;

}

std::vector<double> doubleVectorAddition(const std::vector<double> &vec1, const std::vector<double> &vec2)
{
	assertm(vec1.size() == vec2.size(), "[USER ERROR]: Vector one and Vector two are not of the same size");
	std::vector<double> result;
	for(int i = 0; i < vec1.size(); i++) result.push_back(vec1[i] + vec2[i]);

	return result;

}

std::vector<double> doubleVectorTimesMatrix(const std::vector<double> &vec, const std::vector<std::vector<double>> &mat)
{

	std::vector<double> result;
	int rowLength = mat[0].size();

	assertm(rowLength == vec.size(), "[USER ERROR]: Vector(1,n) and matrix(m,k) where n != m so multiplication could not be done");

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

std::vector<double> scalarTimesVector(double scalar, std::vector<double> &vec)
{
	std::vector<double> result;
	for (double d : vec)
	{
		result.push_back(scalar * d);
	}

	return result;

}


double doubleVecSum(std::vector<double> inputs)
{
	double result = 0;
	for (double d : inputs)
	{
		result = result + d;
	}

	return result;

}

std::vector<double> DSum(std::vector<double> inputs) // sum derivative on vector
{
	std::vector<double> result;
	for (int i = 0; i < inputs.size(); i++)
	{
		result.push_back(1);
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

