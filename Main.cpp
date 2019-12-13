//=======================================================================
// Finn van Vlaanderen
// A6A
// Profielwerkstuk "Neurale Netwerken"
// Programmeergedeelte
//=======================================================================


// Een geoptimalizeerde library voor lineaire algebra
#include <Eigen/Dense>
#include <Eigen/Core>
#include <omp.h>
#include <chrono>

// Mijn eigen geschreven library voor o.a lineaire algebra, 
// documentatie is geschreven in het Engels omdat het een hobbyproject was.
// Dit project maakte *ooit* gebruik van de llinalg-functies hierin, nu gebruik ik het alleen nog maar
// voor een aantal kleine dingen. 
// Correctie: ik gebruik het niet meer.
#include "ala.h"	

// Recht van het internet afgehaald, is wel zo handig.
// Moest alsnog een paar aanpassingen maken om het te laten werken.
#include "mnist/mnist_reader_less.hpp"

// SDL-library, voor datavisualisatie. 
#include <SDL.h>

#include <string>
#include <fstream>
#include <iomanip>
#include <random>
#include <windows.h>
#include <ctime>

// Hier is de visualisatie-component van het project, alhoewel het niets
// toevoegt aan de theorie achter het profielwerkstuk kunt u het natuurlijk
// wel bekijken.
#include "visual.h"

#define uint unsigned int
typedef Eigen::Matrix<uint, Dynamic, 1> VectorXui;

using namespace Eigen;

static int state;

const int im_width = 28;

const int win_height = 600, win_width = 800;
const char* win_title = "Een interessante titel";

//static visual::Window window = visual::Window(win_title, win_width, win_height);

const int pixelSixe = 16; // Grootte van de 'pixels' die getekend worden.
const int centerY = (win_height / 2) - ((pixelSixe * 28) / 2) - 64;
const int centerX = (win_width / 2) - ((pixelSixe * 28) / 2);

const float epsilon = 1e-6;

struct Dataset
{
	Dataset() {}

	Dataset(MatrixXf _training_images, VectorXui _training_labels,
		MatrixXf _test_images, VectorXui _test_labels)
		: training_images(_training_images), training_labels(_training_labels),
		test_images(_test_images), test_labels(_test_labels) {}

	MatrixXf training_images, test_images;
	VectorXui training_labels, test_labels;
};

void print_mnist_image(VectorXf& data);
Dataset init_mnist();

void InitFileSystem()
{
	CreateDirectory("../data", NULL);
	CreateDirectory("../data/graphdata", NULL);
	CreateDirectory("../data/networkdata", NULL);
}

class Timer
{
public:
	Timer(const char* _event_name)
		: start_time(std::chrono::high_resolution_clock::now()),
		event_name(_event_name) {}

	~Timer()
	{
		auto end_time = std::chrono::high_resolution_clock::now();
		long elapsed = (end_time - start_time) / std::chrono::milliseconds(1);
		std::cout << "[Timer] " << event_name << ": "
			<< elapsed << "ms\n";
	}
private:
	const std::chrono::time_point<std::chrono::steady_clock> start_time;
	const char* event_name;
};

// Nee, met "EigenVector" bedoel ik geen eigenvector.
template<typename T>
Matrix<T, Dynamic, 1> ConvertToEigenVector(std::vector<T>& vec)
{
	return Map<Matrix<T, Dynamic, 1>, Unaligned>(vec.data(), vec.size());
}

MatrixXf ConvertToEigenMatrix(std::vector<std::vector<float>> data)
{
	MatrixXf eMatrix(data.size(), data[0].size());
	for (int i = 0; i < data.size(); ++i)
		eMatrix.row(i) = VectorXf::Map(&data[i][0], data[0].size());
	return eMatrix;
}

// Lees de MNIST-dataset en bewaar alle data in vectoren (labels) en
// matrices (afbeeldingen).
Dataset init_mnist()
{
	std::cout << "[MNIST] Parsing MNIST data..\n";
	auto data = mnist::read_dataset();
	std::cout << "[MNIST] Successfully parsed MNIST data\n";

	std::cout << "[MNIST] Converting images..\n";
	MatrixXf ts_images = ConvertToEigenMatrix(data.test_images);
	MatrixXf tr_images = ConvertToEigenMatrix(data.training_images);
	std::cout << "[MNIST] Successfully converted images\n";

	std::cout << "[MNIST] Converting labels..\n";
	VectorXui ts_labels = ConvertToEigenVector(data.test_labels);
	VectorXui tr_labels = ConvertToEigenVector(data.training_labels);
	std::cout << "[MNIST] Successfully converted labels\n";

	Dataset dataset(tr_images, tr_labels, ts_images, ts_labels);
	return dataset;
}

// Print een afbeelding uit de mnist-dataset in de console, 
// uitgedrukt in de waarden van elke pixel.
// \param r De hoeveelste afbeelding die gelezen moet worden.
// \param im_size De breedte/lengte van de afbeeldingen.
// \param data De afbeeldingsmatrix om uit te lezen.
void print_mnist_image(VectorXf& data)
{
	for (int i = 0; i < data.rows(); i++)
	{
		if (i % 28 == 0) std::cout << "\n";
		std::cout << (uint)(data(i) * 255) << " ";
	}
	std::cout << "\n";
}

class NetworkFunction
{
public:

	NetworkFunction(const std::string _name) { name = _name; }

	std::string GetName() const
	{
		return name;
	}
private:
	std::string name = "NaN";
};

class CostFunction : public NetworkFunction
{
public:

	CostFunction(const std::string name)
		: NetworkFunction(name) {}

	virtual float GetCost(VectorXf& a, VectorXf& y) const = 0;
	virtual float GetCost(MatrixXf& a, MatrixXf& y) const = 0;
	
	virtual VectorXf GetDeltaL(VectorXf& a, VectorXf& y, VectorXf& other) const = 0;
	virtual MatrixXf GetDeltaL(MatrixXf& a, MatrixXf& y, MatrixXf& other) const = 0;
};

class ActivationFunction : public NetworkFunction
{
public:

	ActivationFunction(const std::string name)
		: NetworkFunction(name){}

	virtual float Evaluate(float x) const = 0;
	virtual float EvaluateDerivative(float x) const = 0;

	virtual void Evaluate(VectorXf& container, VectorXf& x) const = 0;
	virtual void EvaluateDerivative(VectorXf& container, VectorXf& x) const = 0;

	virtual void Evaluate(MatrixXf& container, MatrixXf& x) const = 0;
	virtual void EvaluateDerivative(MatrixXf& container, MatrixXf& x) const = 0;
};

class RegTerm : public NetworkFunction
{
public:

	RegTerm(const std::string name)
		: NetworkFunction(name){}

	virtual float Evaluate(float lambda, int n, std::vector<MatrixXf>& w) = 0;
	virtual MatrixXf EvaluateDerivative(float lambda, int n, MatrixXf& w) = 0;
};

class L2Regulariser : public RegTerm
{
public:
	
	L2Regulariser()
		: RegTerm("L2"){}

	float Evaluate(float lambda, int n, std::vector<MatrixXf>& w)
	{
		float s = 0;
		for (int l = 0; l < w.size(); l++)
		{
			s += Eigen::pow(w[l].array(), 2).sum();
		}
		return (s * lambda) / (2 * n);
	}

	// Dit is eigenlijk niet de afgeleide van de L2-term, maar er kan op deze manier
	// wel één matrixvermenigvuldigen bespaard worden voor optimalisatie. 
	MatrixXf EvaluateDerivative(float lambda, int n, MatrixXf& w)
	{
		return (lambda / n) * w;
	}
};

class L1Regulariser : public RegTerm
{
public:
	L1Regulariser()
		: RegTerm("L1") {}

	float Evaluate(float lambda, int n, std::vector<MatrixXf>& w)
	{
		float s = 0;
		for (int l = 0; l < w.size(); l++)
		{
			s += Eigen::abs(w[l].array()).sum();
		}
		return (s * lambda) / n;
	}

	MatrixXf EvaluateDerivative(float lambda, int n, MatrixXf& w)
	{
		return (lambda / n) * Eigen::sign(w.array());
	}
};

class SigmoidActivation : public ActivationFunction
{
public:

	SigmoidActivation()
		: ActivationFunction("sigmoid"){}

	float Evaluate(float x) const
	{
		return 1 / (1 + std::exp(-x));
	}

	float EvaluateDerivative(float x) const
	{
		float temp = Evaluate(x);
		return temp * (1 - temp);
	}

	void Evaluate(VectorXf& container, VectorXf& x) const
	{
		container = 1 / (VectorXf::Constant(x.rows(), 1).array() 
			+ Eigen::exp(-x.array()));
	}
	void EvaluateDerivative(VectorXf& container, VectorXf& x) const
	{
		VectorXf temp = 1 / (VectorXf::Constant(x.rows(), 1).array() 
			+ Eigen::exp(-x.array()));
		container = temp.cwiseProduct(VectorXf::Constant(x.rows(), 1) - temp);
	}

	void Evaluate(MatrixXf& container, MatrixXf& x) const
	{
		container = 1 / (MatrixXf::Constant(x.rows(), x.cols(), 1).array() 
			+ Eigen::exp(-x.array()));
	}
	void EvaluateDerivative(MatrixXf& container, MatrixXf& x) const
	{
		MatrixXf temp = 1 / (MatrixXf::Constant(x.rows(), x.cols(), 1).array() 
			+ Eigen::exp(-x.array()));
		container = temp.cwiseProduct(MatrixXf::Constant(x.rows(), x.cols(), 1) - temp);
	}
};

class TanhActivation : public ActivationFunction
{
public:

	TanhActivation()
		: ActivationFunction("tanh") {}

	float Evaluate(float x) const
	{
		return 2 / (1 + std::exp(-2 * x)) - 1;
	}

	float EvaluateDerivative(float x) const
	{
		float temp = Evaluate(x);
		return 1 - (temp * temp);
	}

	void Evaluate(VectorXf& container, VectorXf& x) const
	{
		container = 2 / (1 + Eigen::exp(-2 * x.array()).array()) - 1;
	}
	void EvaluateDerivative(VectorXf& container, VectorXf& x) const
	{
		VectorXf temp = 2 / (1 + Eigen::exp(-2 * x.array()).array()) - 1;
		container = 1 - (temp.cwiseProduct(temp)).array();
	}

	void Evaluate(MatrixXf& container, MatrixXf& x) const
	{
		container = 2 / (1 + Eigen::exp(-2 * x.array()).array()) - 1;
	}
	void EvaluateDerivative(MatrixXf& container, MatrixXf& x) const
	{
		MatrixXf temp = 2 / (1 + Eigen::exp(-2 * x.array()).array()) - 1;
		container = 1 - (temp.cwiseProduct(temp)).array();
	}
};

class ReLUActivation : public ActivationFunction
{
public:

	ReLUActivation()
		: ActivationFunction("relu") {}

	float Evaluate(float x) const
	{
		return max(0.0f, x);
	}

	float EvaluateDerivative(float x) const
	{
		return x > 0 ? 1 : 0;
	}

	void Evaluate(VectorXf& container, VectorXf& x) const
	{
		container = x.cwiseMax(0);
	}
	void EvaluateDerivative(VectorXf& container, VectorXf& x) const
	{
		container = x.array().cwiseMax(0) / x.array();
	}

	void Evaluate(MatrixXf& container, MatrixXf& x) const
	{
		container = x.cwiseMax(0);
	}
	void EvaluateDerivative(MatrixXf& container, MatrixXf& x) const
	{
		container = x.array().cwiseMax(0) / x.array();
	}
};

class LReLUActivation : public ActivationFunction
{
public:

	LReLUActivation()
		: ActivationFunction("leaky relu") {}

	float Evaluate(float x) const
	{
		return (x > 0 ?  x : 0.01f * x);
	}

	float EvaluateDerivative(float x) const
	{
		return x > 0 ? 1 : -0.01f;
	}

	void Evaluate(VectorXf& container, VectorXf& x) const
	{
		container = x.cwiseMax(0.01f * x);
	}
	void EvaluateDerivative(VectorXf& container, VectorXf& x) const
	{
		container = x.cwiseSign().cwiseMax(0.01f);
	}

	void Evaluate(MatrixXf& container, MatrixXf& x) const
	{
		container = x.cwiseMax(0.01f * x);
	}
	void EvaluateDerivative(MatrixXf& container, MatrixXf& x) const
	{
		container = x.cwiseSign().cwiseMax(0.01);
	}
};

int t = 0;

class SoftmaxActivation : public ActivationFunction
{
public:

	SoftmaxActivation()
		: ActivationFunction("softmax") {}

	float Evaluate(float x) const
	{
		return 0; // Softmax voor één waarde kan niet gedefinieerd worden.
	}

	float EvaluateDerivative(float x) const
	{
		return 0; // Softmax voor één waarde kan niet gedefinieerd worden. 
	}

	void Evaluate(VectorXf& container, VectorXf& x) const
	{
		VectorXf expx = x.array().exp();
		container = expx / expx.sum();
	}
	void EvaluateDerivative(VectorXf& container, VectorXf& x) const
	{
		VectorXf expx = x.array().exp();
		expx /= expx.sum();
		container = expx.cwiseProduct(VectorXf::Constant(x.rows(), 1) - expx);
	}

	void Evaluate(MatrixXf& container, MatrixXf& x) const
	{
		MatrixXf expx = x.array().exp();
		container = expx.array().rowwise() / (expx.colwise().sum().array() + epsilon);
	}

	void EvaluateDerivative(MatrixXf& container, MatrixXf& x) const
	{
		MatrixXf expx = x.array().exp();
		expx = expx.array().rowwise() / (expx.colwise().sum().array() + epsilon);
		container = expx.cwiseProduct(MatrixXf::Constant(x.rows(), x.cols(), 1) - expx);
	}
};

class CrossentropyCost : public CostFunction
{
public:

	CrossentropyCost()
		: CostFunction("crossentropy") {};

	float GetCost(VectorXf& a, VectorXf& y) const
	{
		VectorXf ones = VectorXf::Constant(a.rows(), 1);
		return (-(y.array() * a.array().log()) - ((ones.array() - y.array()) 
			* (ones.array() - a.array()).log())).sum();
	}
	VectorXf GetDeltaL(VectorXf& a, VectorXf& y, VectorXf& other) const
	{
		return a - y;
	}

	float GetCost(MatrixXf& A, MatrixXf& Y) const
	{
		MatrixXf onesCol = MatrixXf::Constant(A.cols(), 1, 1);
		MatrixXf ones = MatrixXf::Constant(A.rows(), A.cols(), 1);
		return (((-(Y.array() * A.array().log()) - ((ones - Y).array() 
		* (ones - A).array().log())).colwise().sum()).matrix() * onesCol)
		.operator()(0,0);
	}
	MatrixXf GetDeltaL(MatrixXf& A, MatrixXf& Y, MatrixXf& other) const
	{
		return A - Y;
	}
};

class QuadraticCost : public CostFunction
{
public:
	QuadraticCost()
		: CostFunction("quadratic") {};

	float GetCost(VectorXf& a, VectorXf& y) const
	{
		VectorXf toSquare = a - y;
		return toSquare.dot(toSquare);
	}
	VectorXf GetDeltaL(VectorXf& a, VectorXf& y, VectorXf& other) const
	{
		return (a - y).cwiseProduct(other);
	}

	float GetCost(MatrixXf& A, MatrixXf& Y) const
	{
		MatrixXf onesRow = MatrixXf::Constant(1, A.rows(), 1);
		MatrixXf toSquare = A - Y;
		return ((onesRow * toSquare.cwiseProduct(toSquare)) * onesRow.transpose()).operator()(0,0);
	}
	MatrixXf GetDeltaL(MatrixXf& A, MatrixXf& Y, MatrixXf& other) const
	{
		return (A - Y).cwiseProduct(other);
	}
};

class LoglikelihoodCost : public CostFunction
{
public:

	LoglikelihoodCost()
		: CostFunction("log-likelihood") {};

	float GetCost(VectorXf& a, VectorXf& y) const
	{
		VectorXf::Index maxIndex;
		int x = y.array().maxCoeff(&maxIndex);
		return -log(a(maxIndex));
	}
	VectorXf GetDeltaL(VectorXf& a, VectorXf& y, VectorXf& other) const
	{
		return a - y;
	}

	float GetCost(MatrixXf& A, MatrixXf& Y) const
	{
		VectorXf::Index maxIndex;
		float s = 0;
		for (int i = 0; i < A.cols(); i++)
		{
			int x = Y.col(i).array().maxCoeff(&maxIndex);
			s += -log(A(maxIndex, i));
		}
		return s;
	}
	MatrixXf GetDeltaL(MatrixXf& A, MatrixXf& Y, MatrixXf& other) const
	{
		return A - Y;
	}
};

const CostFunction* quadratic_cost = new QuadraticCost();
const CostFunction* crossentropy_cost = new CrossentropyCost();
const CostFunction* loglikelihood_cost = new LoglikelihoodCost();
const ActivationFunction* relu_activation = new ReLUActivation();
const ActivationFunction* lrelu_activation = new LReLUActivation();
const ActivationFunction* sigmoid_activation = new SigmoidActivation();
const ActivationFunction* softmax_activation = new SoftmaxActivation();
const ActivationFunction* tanh_activation = new TanhActivation();
RegTerm* L2_regulariser = new L2Regulariser();
RegTerm* L1_regulariser = new L1Regulariser();

// De implementatie van het neurale netwerk. 
class Network
{
public:
	Network(std::vector<int> layer_args, const CostFunction* cost_function, const ActivationFunction* activation_function);
	Network(std::vector<int> layer_args, const CostFunction* cost_function, 
		const ActivationFunction* activation_function, const ActivationFunction* L_activation_function);
	Network(Dataset data);
	~Network();

	friend class CostFunction;
	friend class ActivationFunction;

	void FeedForward(int x, bool is_test);
	void Batch_FeedForward(int x, bool is_test);
	int Classify(int x);

	Network* AddLayer(int n_neurons, const ActivationFunction* _activation_function);
	Network* SetRegulariser(RegTerm* reg_term, float lambda);
	Network* SetDropout(float dropout_p);
	Network* SetAccuracyTracking();
	void UpdateDropoutMasks();
	Network* SetOptimizer(int optimizer_id, float beta1, float beta2);
	Network* SetOptimizer(int optimizer_id, float beta1);
	Network* SetCostFunction(const CostFunction* _cost_function);

	void ExpandData(int start, int end);

	void finalize_init();

	VectorXf GetExpectedLabel(int index);
	MatrixXf GetExpectedLabels(int index);

	void Train(int batch_size, int epochs, float eta);

	std::vector<MatrixXf> w;
	std::vector<VectorXf> a, b, z, deltas, expectedResults, r;

	// Voor als het netwerk gevectoriseert is
	std::vector<MatrixXf> A, B, Z, Deltas;

	std::vector<VectorXf> grad_b, cache_b, cache_b_2;
	std::vector<MatrixXf> grad_w, cache_w, cache_w_2;

	const enum optimizers{
		none, 
		momentum_optimizer, 
		adagrad_optimizer,
		adadelta_optimizer,
		adam_optimizer,
		rmsprop_optimizer
	};

private:
	template<typename T>
	void WriteToCSV(std::vector<T>& data);

	const CostFunction* cost_function;
	std::vector<const ActivationFunction*> activation_functions;
	RegTerm* reg_term;

	Dataset dataset;

	std::vector<int> layerNeurons; // Aantal neuronen per verborgen laag.
	std::vector<int> accuracy_buffer;
	
	float error = 0;
	float beta1 = 0, beta2 = 0, lambda = 0, dropout_p = 0.5;
	bool stop_training = 0, do_regularisation = 0, do_dropout = 0;
	bool track_accuracy = 0;
	int optimizer_id = 0, batch_size;
};

Network* Network::SetCostFunction(const CostFunction* _cost_function)
{
	this->cost_function = _cost_function;
	return this;
}

template<typename T>
void Network::WriteToCSV(std::vector<T>& data)
{
	time_t rawtime;
	struct tm* timeinfo;
	char buffer[80];
	time(&rawtime);
	timeinfo = localtime(&rawtime);
	strftime(buffer, sizeof(buffer), 
		"-%d-%m-%Y_%H.%M.%S", timeinfo);
	std::string time_str(buffer);
	std::string path("../data/graphdata/accuracy_epochs");
	path.append(time_str);
	path.append(".csv");
	std::cout << path << "\n";

	std::ofstream file(path);
	file << "Epoch" << "," << "Accuracy" << std::endl;
	for (int i = 0; i < data.size(); i++)
		file  << i << " " << "," << data[i] << std::endl;
	file.close();
	std::cout << "Successfully wrote data to csv\n";
}

Network* Network::SetAccuracyTracking()
{
	track_accuracy = 1;
	return this;
}

void Network::ExpandData(int start, int end)
{
	Timer expand_timer = Timer("expansion_time");

	int oldSize = dataset.training_images.rows();

	Eigen::initParallel();
	if (end > dataset.training_images.rows())
	{
		dataset.training_images.conservativeResize(end, NoChange);
		dataset.training_labels.conservativeResize(end);
	}
	
	std::random_device randomness_device{};
	std::mt19937 pseudorandom_generator{ randomness_device() };
	std::normal_distribution<float> distr(3.14159 / 18, 3.14159 / 30); // 1/18pi, 1/30pi voor 10 en 6 graden
	std::normal_distribution<float> distr2(0, 1);
	float phi = distr(pseudorandom_generator);

	MatrixXf image = MatrixXf::Zero(28, 28);

	// Gaussiaanse convolutiekernel
	Matrix3f kernel;
	kernel << 1, 2, 1, 2, 4, 2, 1, 2, 1;

	int dim = kernel.rows();
	float factor = 1 / 16.0f;

	phi = 3.141592f / 18.0f;
	#pragma omp parallel for num_threads(8)
	for (int i = start; i < end; i++)
	{
		VectorXf x = dataset.training_images.row(i % oldSize);
		VectorXf xPrime = VectorXf::Zero(784);
		
		for (int xIndex = 0; xIndex < dataset.training_images.cols(); xIndex++)
		{
			int u = xIndex % 28;
			int v = std::floor(xIndex / 28);
			float c = x(xIndex);
			
			image(u, v) = c;
		}
		for (int j = dim; j < 28 - dim; j++)
		{
			for (int k = dim; k < 28 - dim; k++)
			{
				xPrime(28 * k + j) = (kernel * image.block(j - 1, k - 1, dim, dim))
					.sum() * factor;
			}
		}
	
		/*
		// Converteer de afbeeldingsvector naar cartesische coördinaten, vermenigvuldig elk punt
		// met een rotatiematrix en converteer terug naar de afbeeldingsvector.
		Matrix2f R;
		R << cos(phi), sin(phi), -sin(phi), cos(phi);

		Vector2f transform;
		transform << 1, 1;
		//std::cout << transform << "\n\n";

		for (int xIndex = 0; xIndex < 784; xIndex++)
		{
			if (x(xIndex) > 0)
			{
				// Psi(x_i)
				Vector2f v((xIndex % 28), (28 - std::floor(xIndex / 28)));
				float c = x(xIndex);

				// Pas de transformatie toe
				v = v + transform;
				
				// Psi^-1(x_i)
				if ((v(0) < 28 && v(0) >= 0) && (v(1) < 28 && v(1) >= 0))
				{
					int xPrimeIndex = std::floor((784 - (28 * v(1)))) + v(0);
					xPrime(xPrimeIndex) = c;
				}
			}
		}
		*/
		if (i > oldSize)
		{
			dataset.training_images.row(i) = xPrime;
			dataset.training_images.row(i) = dataset.training_labels.row(i % oldSize);
		}
		else
		{
			dataset.training_images.row(i) = xPrime;
		}
	}
}

MatrixXf Network::GetExpectedLabels(int index)
{
	MatrixXf Y(10, batch_size);

	for (int i = 0; i < batch_size; i++)
	{
		Y.col(i) = GetExpectedLabel(index + i);
	}
	return Y;
}

void Network::UpdateDropoutMasks()
{
	std::random_device randomness_device{};
	std::mt19937 pseudorandom_generator{ randomness_device() };
	std::bernoulli_distribution distr(dropout_p);
	
	for (int l = 0; l < layerNeurons.size() - 1; l++)
	{
		for (int i = 0; i < r[l].size(); i++)
		{
			/*
			if(l != 0)
				r[l](i) = distr(pseudorandom_generator);
			else
				r[l](i) = 1;
			*/
			r[l](i) = distr(pseudorandom_generator);
		}
	}
}

Network* Network::SetDropout(float _dropout_p)
{
	do_dropout = 1;
	dropout_p = _dropout_p;

	std::random_device randomness_device{};
	std::mt19937 pseudorandom_generator{ randomness_device() };
	std::bernoulli_distribution distr(dropout_p);

	for (int l = 0; l < layerNeurons.size() - 1; l++)
	{
		VectorXf dropout_masks = VectorXf::Zero(layerNeurons[l]);
		for (int i = 0; i < dropout_masks.rows(); i++)
		{
			if (l != 0)
				dropout_masks(i) = distr(pseudorandom_generator) + epsilon;
			else
				dropout_masks(i) = 1;
		}
		r.push_back(dropout_masks);
		std::cout << r[l] << "\n";
	}
	return this;
}

Network* Network::SetOptimizer(int _optimizer_id, float _beta1)
{
	optimizer_id = _optimizer_id;
	beta1 = _beta1;
	return this;
}

Network* Network::SetOptimizer(int _optimizer_id, float _beta1, float _beta2)
{
	optimizer_id = _optimizer_id;
	beta1 = _beta1;
	beta2 = _beta2;
	return this;
}

Network* Network::SetRegulariser(RegTerm* _reg_term, float _lambda)
{
	reg_term = _reg_term;
	lambda = _lambda;
	do_regularisation = 1;
	return this;
}

VectorXf Network::GetExpectedLabel(int index)
{
	return expectedResults[dataset.training_labels(index)];
}

void Network::finalize_init()
{
	std::cout << "[Network|Init] Initializing network values..\n";

	float mu = 0, sigma = 1 / std::sqrt(layerNeurons[0]);

	std::random_device randomness_device{};
	std::mt19937 pseudorandom_generator{ randomness_device() };
	std::normal_distribution<float> weight_distr(mu, sigma);
	std::normal_distribution<float> bias_distr(0, 1);

	for (size_t l = 0; l < layerNeurons.size(); l++)
	{
		MatrixXf zero = MatrixXf::Zero(layerNeurons[l], batch_size);
		A.push_back(zero);
		Z.push_back(zero);
		VectorXf zero_vec = VectorXf::Zero(layerNeurons[l]);
		a.push_back(zero_vec);
		z.push_back(zero_vec);

		if (l > 0)
		{
			deltas.push_back(zero_vec);
			Deltas.push_back(zero);

			MatrixXf w_zero = MatrixXf::Zero(layerNeurons[l], layerNeurons[l - 1]);
			VectorXf b_zero = VectorXf::Zero(layerNeurons[l]);
			MatrixXf weight = w_zero;
			VectorXf bias = b_zero;

			for (int i = 0; i < weight.rows(); i++)
			{
				for (int j = 0; j < weight.cols(); j++)
				{
					weight(i, j) = weight_distr(pseudorandom_generator);
				}
			}

			for (int i = 0; i < bias.rows(); i++)
			{
				bias(i) = bias_distr(pseudorandom_generator);
			}

			w.push_back(weight);
			cache_w.push_back(w_zero);
			cache_w_2.push_back(w_zero);

			b.push_back(bias);
			cache_b.push_back(b_zero);
			cache_b_2.push_back(b_zero);

			grad_b.push_back(b_zero);
			grad_w.push_back(w_zero);
		}
	}

	for (int i = 0; i < 10; i++)
	{
		VectorXf expected_i = VectorXf::Zero(10);
		expected_i(i) = 1;
		expectedResults.push_back(expected_i);
	}
	std::cout << "[Network|Init] Successfully initialized network values\n";
}

// Voeg een laag toe aan het netwerk.
// \param n_neurons Aantal neuronen van de verborgen laag.
Network* Network::AddLayer(int n_neurons, const ActivationFunction* _activation_function)
{
	layerNeurons.push_back(n_neurons);
	activation_functions.push_back(_activation_function);
	std::cout << "[Network|Init] Adding " << _activation_function->GetName() << " layer with " << n_neurons << " neurons..\n";
	return this;
}

Network::Network(Dataset _data) 
: dataset(_data) {}

// \param input_neurons Aantal neuronen in de inputlaag (aantal pixels in afbeelding)
// \param output_neurons Aantal neuronen in de outputlaag (aantal verschillende te herkennen getallen)
Network::Network(std::vector<int> layer_args, const CostFunction* _cost_function, const ActivationFunction* _activation_function)
	: cost_function(_cost_function)
{
	for (int n : layer_args)
		AddLayer(n, _activation_function);
}

Network::Network(std::vector<int> layer_args, const CostFunction* _cost_function,
	const ActivationFunction* _activation_function, const ActivationFunction* _L_activation_function)
	: cost_function(_cost_function)
{
	for (int i = 0; i < layer_args.size(); i++)
	{
		if (i == layer_args.size() - 1)
		{
			AddLayer(layer_args[i], _L_activation_function);
		}
		else
		{
			AddLayer(layer_args[i], _activation_function);
		}
	}
}

Network::~Network() { delete cost_function; for (auto f : activation_functions) delete f; }

void Network::FeedForward(int x, bool is_test)
{
	a[0] = (is_test ? dataset.test_images.row(x) : dataset.training_images.row(x));
	for (int l = 1; l < layerNeurons.size(); l++)
	{
		if (do_dropout && is_test)
		{
			z[l] = (w[l - 1] * dropout_p) * a[l - 1] + b[l - 1];
		}
		else
		{
			if (do_dropout)
			{
				z[l] = (w[l - 1] * a[l - 1].cwiseProduct(r[l - 1])) + b[l - 1];
			}
			else
			{
				z[l] = w[l - 1] * a[l - 1] + b[l - 1]; // Feed-forward voor elke laag.
			}
		}

		activation_functions[l]->Evaluate(a[l], z[l]);
	}
}

void Network::Batch_FeedForward(int x, bool is_test)
{
	for (int i = 0; i < batch_size; i++)
	{
		A[0].col(i) = dataset.training_images.row(x + i);
	}
	for (int l = 1; l < layerNeurons.size(); l++)
	{
		if (do_dropout && is_test)
		{
			Z[l] = ((w[l - 1] * dropout_p) * A[l - 1]).colwise() + b[l - 1];
		}
		else
		{
			if (do_dropout && l != layerNeurons.size() - 1)
			{
				Z[l] = (w[l - 1] * (A[l - 1].array().colwise() * r[l - 1].array()).matrix()).colwise() + b[l - 1];
			}
			else
			{
				Z[l] = (w[l - 1] * A[l - 1]).colwise() + b[l - 1]; // Feed-forward voor elke laag.
			}
		}

		activation_functions[l]->Evaluate(A[l], Z[l]);
	}
}

int Network::Classify(int x)
{
	FeedForward(x, 1);
	VectorXf::Index maxIndex;
	int i = a[layerNeurons.size() - 1].array().maxCoeff(&maxIndex);
	return maxIndex;
}

//typedef std::chrono::time_point<std::chrono::steady_clock> Timer;

void Network::Train(int batch_size, int num_epochs, float eta)
{
	this->batch_size = batch_size;
	finalize_init();

	std::cout << "[Network] Network has started training\n";
	int layers = layerNeurons.size();

	srand(time(NULL));
	int epoch = 0;

	MatrixXf derivatives;
	MatrixXf Y;

	MatrixXf delta_w;
	VectorXf delta_b;

	// Breid de dataset kunstmatig uit; transformeer afbeeldingen x\in[0,10000], vergroot de dataset met 
	// 10k en vul de nieuwe ruimte met de getransformeerde afbeeldingen.
	//ExpandData(60000, 70000);

	//Timer batch_time_begin, batch_time_end;
	Timer training_timer = Timer("training_time");
	while(epoch < num_epochs && !stop_training)
	{
		// De dataset moet gepermuteerd worden voor stochastic gradient descent,
		// hiervoor maken we een willekeurige permutatiematrix p om de datamatrix mee te vermenigvuldigen.
		//std::cout << "[Network|Training] Shuffling data matrix..\n";
		Eigen::PermutationMatrix<Dynamic, Dynamic> p(dataset.training_images.rows());
		p.setIdentity();
		std::random_shuffle(p.indices().data(), p.indices().data() + p.indices().size());
		dataset.training_images = p * dataset.training_images;
		dataset.training_labels = p * dataset.training_labels;
		//std::cout << "[Network|Training] Data matrix has been shuffled\n";

		for (int l = 1; l < layers; l++)
		{
			cache_w[l - 1].fill(0);
			cache_w_2[l - 1].fill(0);
			cache_b[l - 1].fill(0);
			cache_b_2[l - 1].fill(0);
		}
		
		Timer epoch_timer = Timer("epoch_time");
		std::cout << "\n";

		// Doe Stochastic Gradient Descent
		for (int set_index = 0; set_index + batch_size < dataset.training_images.rows(); set_index += batch_size)
		{
			t = (set_index / batch_size) + 1;

			if (set_index % 10000 == 0)
			{
				error = 0;
			}
			
			// Feed-forward
			Batch_FeedForward(set_index, 0);
			Y = GetExpectedLabels(set_index);

			// Backpropagation
			for (int l = layers - 1; l > 0; l--)
			{
				activation_functions[l]->EvaluateDerivative(derivatives, Z[l]);
				if (l == layers - 1)
					Deltas[l - 1] = cost_function->GetDeltaL(A[layers - 1], Y, derivatives);
				else
					Deltas[l - 1] = (w[l].transpose() * Deltas[l]).cwiseProduct(derivatives);
				
				grad_b[l - 1] = Deltas[l - 1].rowwise().sum() / batch_size;

				for (int i = 0; i < batch_size; i++)
				{
					grad_w[l - 1] += Deltas[l - 1].col(i) * A[l - 1].col(i).transpose();
				}
				grad_w[l - 1] /= batch_size;
			}
				
			// Bereken de fout van deze batch.
			if (set_index % 10000 == 0)
			{
				error = cost_function->GetCost(A[layers - 1], Y);
				if (do_regularisation)
					error += reg_term->Evaluate(lambda, dataset.training_images.size(), w);
			}

			// Update de waarden met de gradiënt
			for (int l = 0; l < layers - 1; l++)
			{	
				switch (optimizer_id)
				{
				case momentum_optimizer:
				{
					cache_w[l] = beta1 * cache_w[l] + eta * grad_w[l];
					w[l] -= cache_w[l];

					cache_b[l] = beta1 * cache_b[l] + eta * grad_b[l];
					b[l] -= cache_b[l];
				}
				break;
				case adagrad_optimizer:
				{
					cache_w[l] += grad_w[l].cwiseProduct(grad_w[l]);
					cache_b[l] += grad_b[l].cwiseProduct(grad_b[l]);

					w[l] -= eta * 
						grad_w[l].cwiseProduct((
						(cache_w[l].array() + epsilon)
						.rsqrt()).matrix());

					b[l] -= eta * 
						grad_b[l].cwiseProduct((
						(cache_b[l].array() + epsilon)
						.rsqrt()).matrix());
				}
				break;
				case rmsprop_optimizer:
				{
					/* 
					Ik weet nu dat:
					De cache is op t-1 nog niet NaN, zelfde geldt voor rsqrt.
					Op t is de cache wel NaN.
					We kunnen concluderen dat de omslag zich moet bevinden 
					in de cache-update op tijdstip t, en NIET in de wortel, aangezien de cache EN
					de wortel NaN zijn op t. 
					Conclusie: De gradiënt nullt. 
					*/

					/*
					if (epoch >= 2 && t > 3000)
					{
						std::cout << "cache voor update:\n" << cache_b[l] << "\n\n";
						std::cout << "gradient:\n" << grad_b[l] << "\n\n";
					}
					*/

					cache_w[l] = beta1 * cache_w[l] + (1 - beta1) 
						* (grad_w[l].cwiseProduct(grad_w[l]));

					cache_b[l] = beta1 * cache_b[l] + (1 - beta1) 
						* (grad_b[l].cwiseProduct(grad_b[l]));

					/*
					if (epoch >= 2 && t > 3000)
					{
						std::cout << "cache na update:\n" << cache_b[l] << "\n\n";
					}
					*/

					delta_w = eta * grad_w[l].cwiseProduct((
						cache_w[l].array() + epsilon)
						.rsqrt().matrix());

					delta_b = eta * grad_b[l].cwiseProduct((
						cache_b[l].array() + epsilon)
						.rsqrt().matrix());

					w[l] -= delta_w;

					b[l] -= delta_b;

					/*
					if (epoch >= 3)
					{
						std::cout << "noemer: " << cache_b[l].array() + epsilon << "\n\n";
						std::cout << "rsqrt: " << (cache_b[l].array() + epsilon).rsqrt() << "\n\n";
					}

					if (isnan((cache_b[l].array() + epsilon).rsqrt().minCoeff()))
					{
						std::cout << "sqrt oops op t=" << t << "\n";
					}
					*/
				}
				break;
				case adadelta_optimizer:
				{
					//=====================//
					cache_w[l] =
						beta1 * cache_w[l] +
						(1 - beta1) * grad_w[l].cwiseProduct(grad_w[l]);

					delta_w = -Eigen::sqrt((cache_w_2[l].array() + epsilon)
						/ (cache_w[l].array() + epsilon)).matrix().cwiseProduct(grad_w[l]);

					cache_w_2[l] =
						beta1 * cache_w_2[l] +
						(1 - beta1) * delta_w.cwiseProduct(delta_w);

					w[l] += delta_w;
					//=====================//
					cache_b[l] =
						beta1 * cache_b[l] +
						(1 - beta1) * grad_b[l].cwiseProduct(grad_b[l]);

					delta_b = -Eigen::sqrt((cache_b_2[l].array() + epsilon)
						/ (cache_b[l].array() + epsilon)).matrix().cwiseProduct(grad_b[l]);

					cache_b_2[l] = 
						beta1 * cache_b_2[l] + 
						(1 - beta1) * delta_b.cwiseProduct(delta_b);

					b[l] += delta_b;
					//=====================//
				}
				break;
				case adam_optimizer:
				{
					//=====================//
					cache_w[l] = beta1 * cache_w[l] + (1 - beta1) * grad_w[l];

					cache_w_2[l] = beta2 * cache_w_2[l] + 
						(1 - beta2) * (grad_w[l].cwiseProduct(grad_w[l]));

					MatrixXf mw = cache_w[l] / (1.0f - std::pow(beta1, t));
					MatrixXf vw = cache_w_2[l] / (1.0f - std::pow(beta2, t));

					w[l] -= eta * mw.cwiseProduct((vw.array() + epsilon).rsqrt().matrix());
					//=====================//
					cache_b[l] = beta1 * cache_b[l] + (1 - beta1) * grad_b[l];

					cache_b_2[l] = beta2 * cache_b_2[l] +
						(1 - beta2) * (grad_b[l].cwiseProduct(grad_b[l]));

					MatrixXf mb = cache_b[l] / (1.0f - std::pow(beta1, t));
					MatrixXf vb = cache_b_2[l] / (1.0f - std::pow(beta2, t));

					b[l] -= eta * mb.cwiseProduct((vb.array() + epsilon).rsqrt().matrix());
					//=====================//
				}
				break;
				case none:
				{
					w[l] -= eta * grad_w[l];
					b[l] -= eta * grad_b[l];
				}
				break;
				}

				if (do_regularisation)
				{
					w[l] -= eta * reg_term->EvaluateDerivative(lambda, dataset.training_images.rows(), w[l]);
				}
				grad_w[l].fill(0);
				grad_b[l].fill(0);
			}

			if (set_index % 10000 == 0)
			{
				error /= batch_size;
				if (isnan(error)) error = 0;
				std::cout << "[Network|Training] Error: " << error << "\n";
			}
		}

		// Check progress
		int size = dataset.test_images.rows();
		float accuracy = 0;
		for (int i = 0; i < size; i++)
			accuracy += (Classify(i) == dataset.test_labels(i));
		std::cout << "[Network|Training] Epochs trained: " << ++epoch << "\n";
		std::cout << "[Network|Training] Accuracy: " << accuracy << " / " << size << "\n";

		// Bewaar accuracy en epoch, om later te schrijven naar een csv-bestand.
		if (track_accuracy)
		{
			accuracy_buffer.push_back(accuracy);
		}

		if (do_dropout)
			UpdateDropoutMasks();
	}
	
	WriteToCSV(accuracy_buffer);
	std::cout << "[Network] Network has successfully been trained\n";
}

int main(int argc, char **argv)
{
	InitFileSystem();

	//Eigen::initParallel();
	omp_set_num_threads(8);
	Eigen::setNbThreads(0);
	std::cout << "[Debug] Running Eigen on " << Eigen::nbThreads() << " threads\n";
	//state = window.loading_state;
	//visual::DataGrid grid(window, &ts_images, pixelSixe, centerX, centerY); 

	//while (true /*!window.isClosed()*/)
	//{
		//if (state == window.explore_state)
		//{
			// TODO: doe dit op een competente manier
			//grid.draw();
			//int n = network->Classify(ts_images, grid.dataMatrix_row);
			//std::cout << "Het netwerk denkt dat dit een " << n << " is.\n";
			//grid.pollEvents();
		//}

		//window.pollEvents();
		//window.clear();

		//if (state == window.loading_state)
		//{
			//Dataset dataset = init_mnist();

			Network* network = new Network(init_mnist());
			network->SetCostFunction(loglikelihood_cost);
			network->AddLayer(784, lrelu_activation);
			network->AddLayer(32, lrelu_activation);
			network->AddLayer(32, lrelu_activation);
			network->AddLayer(10, softmax_activation);
			network->SetRegulariser(L2_regulariser, 6.0);
			network->SetOptimizer(network->rmsprop_optimizer, 0.9, 0.9);
			network->SetAccuracyTracking();

			network->Train(10, 100, 0.01);

			//grid.init_data();

			//state = window.explore_state;
		//}
	//}
	return 0;
}