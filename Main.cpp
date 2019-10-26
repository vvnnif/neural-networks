//=======================================================================
// Finn van Vlaanderen
// A6A
// Profielwerkstuk "Neurale Netwerken"
// Programmeergedeelte / deelvraag 4
//=======================================================================

// Een geoptimalizeerde library voor lineaire algebra; want 200000 matrixvermenigvuldigingen
// is VEEL.
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

// Hier is de visualisatie-component van het project, alhoewel het niets
// toevoegt aan de theorie achter het profielwerkstuk kunt u het natuurlijk
// wel bekijken.
#include "visual.h"

#define uint unsigned int
typedef Eigen::Matrix<uint, Dynamic, 1> VectorXui;

using namespace Eigen;

static VectorXui tr_labels, ts_labels; // Labels
static Eigen::MatrixXf tr_images, ts_images; // Afbeeldingen

static int state;

const int im_width = 28;

const int win_height = 600, win_width = 800;
const char* win_title = "Een interessante titel";

static visual::Window window = visual::Window(win_title, win_width, win_height);

const int pixelSixe = 16; // Grootte van de 'pixels' die getekend worden.
const int centerY = (win_height / 2) - ((pixelSixe * 28) / 2) - 64;
const int centerX = (win_width / 2) - ((pixelSixe * 28) / 2);

void print_mnist_image(VectorXf& data);
void init_mnist();

class Timer
{
public:
	Timer(const std::string& _event_name)
		: start_time(start_time = std::chrono::high_resolution_clock::now()),
		event_name(_event_name) {}

	~Timer()
	{
		auto end_time = std::chrono::high_resolution_clock::now();
		long elapsed = (end_time - start_time) / std::chrono::milliseconds(1);
		std::cout << "[Timer] " << event_name << ": "
			<< elapsed << "ms\n";
	}
private:
	std::chrono::time_point<std::chrono::steady_clock> start_time;
	const std::string event_name;
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
void init_mnist()
{
	std::cout << "[MNIST] Parsing MNIST data..\n";
	auto dataset = mnist::read_dataset();
	std::cout << "[MNIST] Successfully parsed MNIST data\n";

	std::cout << "[MNIST] Converting images..\n";
	ts_images = ConvertToEigenMatrix(dataset.test_images);
	tr_images = ConvertToEigenMatrix(dataset.training_images);
	std::cout << "[MNIST] Successfully converted images\n";

	std::cout << "[MNIST] Converting labels..\n";
	ts_labels = ConvertToEigenVector(dataset.test_labels);
	tr_labels = ConvertToEigenVector(dataset.training_labels);
	std::cout << "[MNIST] Successfully converted labels\n";
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

class L2Regularisation : public RegTerm
{
public:
	
	L2Regularisation()
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

class L1Regularisation : public RegTerm
{
public:
	L1Regularisation()
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

class ReLUActivation : public ActivationFunction
{
public:

	ReLUActivation()
		: ActivationFunction("ReLU") {}

	float Evaluate(float x) const
	{
		return std::max(0.0f, x);
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
		container = expx.array().rowwise() / expx.colwise().sum().array();
	}
	void EvaluateDerivative(MatrixXf& container, MatrixXf& x) const
	{
		MatrixXf expx = x.array().exp();
		expx = expx.array() / expx.colwise().sum().array();
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
const ActivationFunction* ReLU_activation = new ReLUActivation();
const ActivationFunction* sigmoid_activation = new SigmoidActivation();
const ActivationFunction* softmax_activation = new SoftmaxActivation();
RegTerm* L2_regularisation = new L2Regularisation();
RegTerm* L1_regularisation = new L1Regularisation();

// De implementatie van het neurale netwerk. 
class Network
{
public:
	Network(std::vector<int> layer_args, const CostFunction* cost_function, const ActivationFunction* activation_function);
	Network(std::vector<int> layer_args, const CostFunction* cost_function, 
		const ActivationFunction* activation_function, const ActivationFunction* L_activation_function);
	~Network();

	friend class CostFunction;
	friend class ActivationFunction;

	void FeedForward(int x, bool is_test);
	void Batch_FeedForward(int x, bool is_test);
	int Classify(MatrixXf& data, int x);

	Network* AddLayer(int n_neurons);
	Network* SetRegularisation(RegTerm* reg_term, float lambda);
	Network* SetDropout(float dropout_p);
	void UpdateDropoutMasks();
	Network* SetOptimizer(int optimizer_id, float gamma);

	void ExpandData(int start, int end);

	void finalize_init();

	VectorXf GetExpectedLabel(int index);
	MatrixXf GetExpectedLabels(int index);

	void Train(int batch_size, int epochs, float eta);

	std::vector<MatrixXf> w;
	std::vector<VectorXf> a, b, z, deltas, expectedResults, r;

	// Voor als het netwerk gevectoriseert is
	std::vector<MatrixXf> A, B, Z, Deltas;

	std::vector<VectorXf> gradient_b, vel_b;
	std::vector<MatrixXf> gradient_w, vel_w;

	const enum optimizers{none, momentum_optimizer, adagrad_optimizer};

private:
	const CostFunction* cost_function;
	const ActivationFunction* activation_function, * L_activation_function;
	RegTerm* reg_term;

	std::vector<int> layerNeurons; // Aantal neuronen per verborgen laag.
	
	float error = 0;
	float gamma = 0, lambda = 0, dropout_p = 0.5;
	const float epsilon = 1e-4;
	bool stop_training = 0, do_regularisation = 0, do_dropout = 0, do_vectorization = 0;
	int optimizer_id = 0, batch_size;
};

void Network::ExpandData(int start, int end)
{
	Timer expand_timer = Timer("expansion_time");

	Eigen::initParallel();
	tr_images.conservativeResize(tr_images.rows() + (end - start), NoChange);
	tr_labels.conservativeResize(tr_labels.rows() + (end - start));
	
	std::random_device randomness_device{};
	std::mt19937 pseudorandom_generator{ randomness_device() };
	std::normal_distribution<float> distr(3.141 / 18, 3.141 / 30); // 1/18pi, 1/30pi voor 10 en 6 graden
	float phi = distr(pseudorandom_generator);

	#pragma omp parallel for num_threads(8)
	for (int i = 0; i < (end - start); i++)
	{
		// Converteer de afbeeldingsvector naar cartesische coördinaten, vermenigvuldig elk punt
		// met een rotatiematrix en converteer terug naar de afbeeldingsvector.
		VectorXf x = tr_images.row(start + i);
		VectorXf xPrime = VectorXf::Zero(784);
		Matrix2f R;
		R << cos(phi), -sin(phi), sin(phi), cos(phi);
		for (int xIndex = 0; xIndex < 784; xIndex++)
		{
			if (x(xIndex) > 0)
			{
				// Phi(x_i)
				Vector2f v((xIndex % 28) - 14, (28 - std::floor(xIndex / 28)) - 14);
				float c = x(xIndex);

				// RvA
				v = R * v;
				
				// {Phi}^{-1}(x_i)
				if ((v(0) < 14 && v(0) >= -14) && (v(1) < 14 && v(1) >= -14))
				{
					int xPrimeIndex = std::floor((784 - (28 * (v(1) + 14))) + v(0) + 14);
					xPrime(xPrimeIndex) = c;
				}
			}
		}

		tr_images.row(tr_images.rows() - (end - start) + i) = xPrime;
		tr_labels(tr_labels.rows() - (end - start) + i) = tr_labels(start + i);

		//print_mnist_image(x);
		//std::cout << "\n";
		//print_mnist_image(xPrime);
		//std::cout << "\n";
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
			r[l](i) = distr(pseudorandom_generator);
		}
		std::cout << r[l] << "\n\n";
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
		VectorXf dropout_masks = VectorXf::Constant(layerNeurons[l], 0);
		for (int i = 0; i < dropout_masks.rows(); i++)
		{
			dropout_masks(i) = distr(pseudorandom_generator) + epsilon;
		}
		r.push_back(dropout_masks);
		std::cout << r[l] << "\n\n";
	}
	return this;
}

Network* Network::SetOptimizer(int _optimizer_id, float _gamma)
{
	optimizer_id = _optimizer_id;
	gamma = _gamma;
	return this;
}

Network* Network::SetRegularisation(RegTerm* _reg_term, float _lambda)
{
	reg_term = _reg_term;
	lambda = _lambda;
	do_regularisation = 1;
	return this;
}

VectorXf Network::GetExpectedLabel(int index)
{
	return expectedResults[tr_labels[index]];
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
			vel_w.push_back(w_zero);

			b.push_back(bias);
			vel_b.push_back(b_zero);

			gradient_b.push_back(b_zero);
			gradient_w.push_back(w_zero);
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
Network* Network::AddLayer(int n_neurons)
{
	layerNeurons.push_back(n_neurons);
	std::cout << "[Network|Init] Adding layer with " << n_neurons << " neurons..\n";
	return this;
}

// \param input_neurons Aantal neuronen in de inputlaag (aantal pixels in afbeelding)
// \param output_neurons Aantal neuronen in de outputlaag (aantal verschillende te herkennen getallen)
Network::Network(std::vector<int> layer_args, const CostFunction* _cost_function, const ActivationFunction* _activation_function)
	: cost_function(_cost_function), activation_function(_activation_function), L_activation_function(_activation_function)
{
	for (int n : layer_args)
		AddLayer(n);
}

Network::Network(std::vector<int> layer_args, const CostFunction* _cost_function,
	const ActivationFunction* _activation_function, const ActivationFunction* _L_activation_function)
	: cost_function(_cost_function), activation_function(_activation_function), L_activation_function(_L_activation_function)
{
	for (int n : layer_args)
		AddLayer(n);
}

Network::~Network() { delete cost_function; delete activation_function; }

void Network::FeedForward(int x, bool is_test)
{
	a[0] = (is_test ? ts_images.row(x) : tr_images.row(x));
	for (int l = 1; l < layerNeurons.size(); l++)
	{
		if (do_dropout && is_test)
		{
			z[l] = (w[l - 1] * dropout_p) * a[l - 1] + b[l - 1];
		}
		else
		{
			if (do_dropout && l < layerNeurons.size() - 1)
			{
				z[l] = (w[l - 1] * a[l - 1].cwiseProduct(r[l - 1])) + b[l - 1];
			}
			else
			{
				z[l] = w[l - 1] * a[l - 1] + b[l - 1]; // Feed-forward voor elke laag.
			}
		}

		if (l == layerNeurons.size() - 1)
			L_activation_function->Evaluate(a[l], z[l]);
		else
			activation_function->Evaluate(a[l], z[l]);
	}
}

void Network::Batch_FeedForward(int x, bool is_test)
{
	for (int i = 0; i < batch_size; i++)
	{
		A[0].col(i) = tr_images.row(x + i);
	}
	for (int l = 1; l < layerNeurons.size(); l++)
	{
		if (do_dropout && is_test)
		{
			Z[l] = ((w[l - 1] * dropout_p) * A[l - 1]).colwise() + b[l - 1];
		}
		else
		{
			if (do_dropout && l < layerNeurons.size() - 1)
			{
				Z[l] = (w[l - 1] * A[l - 1].cwiseProduct(r[l - 1])).colwise() + b[l - 1];
			}
			else
			{
				Z[l] = (w[l - 1] * A[l - 1]).colwise() + b[l - 1]; // Feed-forward voor elke laag.
			}
		}

		if (l == layerNeurons.size() - 1)
			L_activation_function->Evaluate(A[l], Z[l]);
		else
			activation_function->Evaluate(A[l], Z[l]);
	}
}

int Network::Classify(MatrixXf& data, int x)
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

	//Timer batch_time_begin, batch_time_end;
	Timer training_timer = Timer("training_time");
	while(epoch < num_epochs && !stop_training)
	{
		// De dataset moet gepermuteerd worden voor stochastic gradient descent,
		// hiervoor maken we een willekeurige permutatiematrix p om de datamatrix mee te vermenigvuldigen.
		//std::cout << "[Network|Training] Shuffling data matrix..\n";
		Eigen::PermutationMatrix<Dynamic, Dynamic> p(tr_images.rows());
		p.setIdentity();
		std::random_shuffle(p.indices().data(), p.indices().data() + p.indices().size());
		tr_images = p * tr_images;
		tr_labels = p * tr_labels; // Vermenigvuldig ook de labels met p.
		//std::cout << "[Network|Training] Data matrix has been shuffled\n";
		
		Timer epoch_timer = Timer("epoch_time");
		std::cout << "\n";
		// Doe Stochastic Gradient Descent
		for (int set_index = 0; set_index + batch_size < tr_images.rows(); set_index += batch_size)
		{
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
				activation_function->EvaluateDerivative(derivatives, Z[l]);
				if (l == layers - 1)
					Deltas[l - 1] = cost_function->GetDeltaL(A[layers - 1], Y, derivatives);
				else
					Deltas[l - 1] = (w[l].transpose() * Deltas[l]).cwiseProduct(derivatives);
				gradient_b[l - 1] = Deltas[l - 1].rowwise().sum() / batch_size;
				for (int i = 0; i < batch_size; i++)
				{
					gradient_w[l - 1] += Deltas[l - 1].col(i) * A[l - 1].col(i).transpose();
				}
				gradient_w[l - 1] /= batch_size;
			}
				
			// Bereken de fout van deze batch.
			if (set_index % 10000 == 0)
			{
				error = cost_function->GetCost(A[layers - 1], Y);
				if (do_regularisation)
				{
					error += reg_term->Evaluate(lambda, tr_images.size(), w);
				}
			}
			
			// Update de waarden met de gradiënt
			for (int l = 1; l < layers; l++)
			{	
				switch (optimizer_id)
				{
				case momentum_optimizer:
				{
					vel_w[l - 1] = (gamma * vel_w[l - 1]) - (eta * gradient_w[l - 1]);
					w[l - 1] += vel_w[l - 1];

					vel_b[l - 1] = (gamma * vel_b[l - 1]) - (eta * gradient_b[l - 1]);
					b[l - 1] += vel_b[l - 1];
				}
				break;
				case adagrad_optimizer:
				{
					w[l - 1] -= (vel_w[l - 1].array() + epsilon).pow(-0.5).matrix().cwiseProduct(gradient_w[l - 1]) * eta;
					b[l - 1] -= (vel_b[l - 1].array() + epsilon).pow(-0.5).matrix().cwiseProduct(gradient_b[l - 1]) * eta;

					vel_w[l - 1] += gradient_w[l - 1].array().pow(2).matrix();
					vel_b[l - 1] += gradient_b[l - 1].array().pow(2).matrix();
				}
				break;
				case none:
				{
					w[l - 1] -= eta * gradient_w[l - 1];
					b[l - 1] -= eta * gradient_b[l - 1];
				}
				break;
				}

				if (do_regularisation)
				{
					w[l - 1] -= eta * reg_term->EvaluateDerivative(lambda, tr_images.rows(), w[l - 1]);
				}
				gradient_w[l - 1].fill(0);
				gradient_b[l - 1].fill(0);
			}

			if (set_index % 10000 == 0)
			{
				error /= batch_size;
				if (isnan(error)) error = 0;
				std::cout << "[Network|Training] Error: " << error << "\n";
			}
		}

		// Check progress
		int size = ts_images.rows();
		float recognised = 0;
		for (int i = 0; i < size; i++)
			recognised += (Classify(ts_images, i) == ts_labels[i]);
		std::cout << "[Network|Training] Epochs trained: " << ++epoch << "\n";
		std::cout << "[Network|Training] Accuracy: " << recognised << " / " << size << "\n";

		srand(time(NULL));
		int start = rand() % 60000;

		if(epoch <= 4)
			//ExpandData(start, start + 10000);

		if (do_dropout)
			UpdateDropoutMasks();
	}
	//std::cout << "[Network] Final error is:" << error << "\n";
	std::cout << "[Network] Network has successfully been trained\n";
}

// Laad het neurale netwerk; hierin worden alle parameters van het netwerk
// vastgesteld. Hoeveelheid lagen, hoeveelheid neuronen in elke laag,
// activatiefunctie, et cetera.
void TrainInParallel(int num_networks)
{
#pragma omp parallel num_threads(num_networks)
	{
		/*
		srand(omp_get_thread_num() << 8);
		float r = ala::GetUniformRandom<float>(0.001, 1);
		std::cout << r << "\n";
		Network* network = new Network({ 784, 32, 10 },
			loglikelihood_cost,
			ReLU_activation,
			softmax_activation);
		
		network->SetRegularisation(L2_regularisation, 5.0);
		network->SetOptimizer(network->momentum_optimizer, 0.9);
		//network->SetDropout(0.5);
		network->Train(10, 100, r);
		*/
	}
}

int main(int argc, char **argv)
{
	//Eigen::initParallel();
	omp_set_num_threads(8);
	std::cout << "[Debug] Running Eigen on " << Eigen::nbThreads() << " threads\n";
	state = window.loading_state;
	visual::DataGrid grid(window, &ts_images, pixelSixe, centerX, centerY); 

	while (!window.isClosed())
	{
		if (state == window.explore_state)
		{
			// TODO: doe dit op een competente manier
			grid.draw();
			//int n = network->Classify(ts_images, grid.dataMatrix_row);
			//std::cout << "Het netwerk denkt dat dit een " << n << " is.\n";
			grid.pollEvents();
		}

		window.pollEvents();
		window.clear();

		if (state == window.loading_state)
		{
			init_mnist();

			// >98%!
			Network* network = new Network({ 784, 64, 10 },
				loglikelihood_cost,
				sigmoid_activation,
				softmax_activation);

			//network->SetRegularisation(L2_regularisation, 3.0);
			network->SetOptimizer(network->momentum_optimizer, 0.9);
			network->SetDropout(0.5);
			network->Train(10, 200, 0.085);

#ifndef _DO_PARALLEL_TRAINING
#define _DO_PARALLEL_TRAINING
#endif

#ifdef _DO_PARALLEL_TRAINING
			//TrainInParallel(2);
#endif

			//grid.init_data();

			state = window.explore_state;
		}
	}
	return 0;
}