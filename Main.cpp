//=======================================================================
// Finn van Vlaanderen
// A6A
// Profielwerkstuk "Neurale Netwerken"
// Programmeergedeelte / deelvraag 4
//=======================================================================

// Een geoptimalizeerde library voor lineaire algebra; want 200000 matrixvermenigvuldigingen
// is VEEL.
#include <Eigen/Dense>

// Mijn eigen geschreven library voor o.a lineaire algebra, 
// documentatie is geschreven in het Engels omdat het een hobbyproject was.
// Dit project maakte *ooit* gebruik van de llinalg-functies hierin, nu gebruik ik het alleen nog maar
// voor een aantal kleine dingen. 
// Correctie: ik gebruik het niet meer.
//#include "ala.h"	

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

void init_network();

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
	virtual VectorXf GetDeltaL(VectorXf& a, VectorXf& y, VectorXf& other) const = 0;

};

class ActivationFunction : public NetworkFunction
{
public:

	ActivationFunction(const std::string name)
		: NetworkFunction(name){}

	virtual float Evaluate(float x) const = 0;
	virtual float EvaluateDerivative(float x) const = 0;

	virtual VectorXf Evaluate(VectorXf& x) const = 0;
	virtual VectorXf EvaluateDerivative(VectorXf& x) const = 0;
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

	VectorXf Evaluate(VectorXf& x) const
	{
		return 1 / (VectorXf::Constant(x.rows(), 1).array() + Eigen::exp(-x.array()));
	}

	VectorXf EvaluateDerivative(VectorXf& x) const
	{
		VectorXf temp = Evaluate(x);
		return temp.cwiseProduct(VectorXf::Constant(x.rows(), 1) - temp);
	}
};

class ReLUActivation : public ActivationFunction
{
public:

	ReLUActivation()
		: ActivationFunction("ReLU") {}

	float Evaluate(float x) const
	{
		return x > 0 ? x : 0;
	}

	float EvaluateDerivative(float x) const
	{
		return x > 0 ? 1 : 0;
	}

	VectorXf Evaluate(VectorXf& x) const
	{
		return x.cwiseMax(0);
	}

	VectorXf EvaluateDerivative(VectorXf& x) const
	{
		return x.array().cwiseMax(0) / x.array();
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

	VectorXf Evaluate(VectorXf& x) const
	{
		VectorXf expx = x.array().exp();
		return expx / expx.sum();
	}

	VectorXf EvaluateDerivative(VectorXf& x) const
	{
		VectorXf temp = Evaluate(x);
		return temp.cwiseProduct(VectorXf::Constant(x.rows(), 1) - temp);
	}
};

class CrossentropyCost : public CostFunction
{
public:

	CrossentropyCost()
		: CostFunction("crossentropy") {};

	float GetCost(VectorXf& a, VectorXf& y) const
	{
		/*
		float s = 0;
		for (int i = 0; i < a.rows(); i++)
		{
			s += (-y(i) * std::log(a(i))) - ((1 - y(i)) * std::log((1 - a(i))));
		}
		return s;
		*/
		VectorXf ones = VectorXf::Constant(a.rows(), 1);
		return -((y.array() * a.array().log()) - ((ones.array() - y.array()) * (ones.array() - a.array()).log())).sum();
	}

	VectorXf GetDeltaL(VectorXf& a, VectorXf& y, VectorXf& other) const
	{
		return a - y;
	}
};

class QuadraticCost : public CostFunction
{
public:
	QuadraticCost()
		: CostFunction("quadratic") {};

	float GetCost(VectorXf& a, VectorXf& y) const
	{
		float s = 0;
		VectorXf toSquare = a - y;
		s += toSquare.dot(toSquare);
		return s;
	}

	VectorXf GetDeltaL(VectorXf& a, VectorXf& y, VectorXf& other) const
	{
		return (a - y).cwiseProduct(other);
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
};

const CostFunction* quadratic_cost = new QuadraticCost();
const CostFunction* crossentropy_cost = new CrossentropyCost();
const CostFunction* loglikelihood_cost = new LoglikelihoodCost();
const ActivationFunction* ReLU_activation = new ReLUActivation();
const ActivationFunction* sigmoid_activation = new SigmoidActivation();
const ActivationFunction* softmax_activation = new SoftmaxActivation();
static RegTerm* L2_regularisation = new L2Regularisation();
static RegTerm* L1_regularisation = new L1Regularisation();

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

	void FeedForward(MatrixXf& data, int x);
	int Classify(MatrixXf& data, int x);

	Network* AddLayer(int n_neurons);
	Network* SetRegularisation(RegTerm* reg_term);

	void finalize_init();

	VectorXf GetExpectedLabel(int index);

	void Train(int batch_size, int epochs, float learning_rate);
	void Train(int batch_size, int epochs, float learning_rate, float lambda);

	std::vector<MatrixXf> w;
	std::vector<VectorXf> a, b, z, deltas, expectedResults;

	std::vector<VectorXf> gradient_b;
	std::vector<MatrixXf> gradient_w;
private:
	const CostFunction* cost_function;
	const ActivationFunction* activation_function, * L_activation_function;
	RegTerm* reg_term;

	std::vector<int> layerNeurons; // Aantal neuronen per verborgen laag.

	float error = 0;
	bool stop_training = 0, do_regularisation = 0;
};

Network* Network::SetRegularisation(RegTerm* _reg_term)
{
	reg_term = _reg_term;
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

	for (size_t i = 0; i < layerNeurons.size(); i++)
	{
		VectorXf zero = VectorXf::Zero(layerNeurons[i]); 
		a.push_back(zero);
		z.push_back(zero);

		if (i > 0)
		{
			deltas.push_back(zero);

			MatrixXf weight = MatrixXf::Constant(layerNeurons[i], layerNeurons[i - 1], 0);
			VectorXf bias = VectorXf::Constant(layerNeurons[i], 0);

			for (int i = 0; i < weight.rows(); i++)
			{
				for (int j = 0; j < weight.cols(); j++)
				{
					weight(i, j) = weight_distr(pseudorandom_generator);
					//std::cout << "[Debug] Initialized weight as " << weight(i, j) << "\n";
				}
			}

			for (int i = 0; i < bias.rows(); i++)
			{
				bias(i) = bias_distr(pseudorandom_generator);
			}

			w.push_back(weight);
			b.push_back(bias);

			//std::cout << randomWeightMatrix << "\n";

			gradient_b.push_back(zero);
			gradient_w.push_back(MatrixXf::Zero(layerNeurons[i], layerNeurons[i - 1]));
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
	finalize_init();
}

Network::Network(std::vector<int> layer_args, const CostFunction* _cost_function,
	const ActivationFunction* _activation_function, const ActivationFunction* _L_activation_function)
	: cost_function(_cost_function), activation_function(_activation_function), L_activation_function(_L_activation_function)
{
	for (int n : layer_args)
		AddLayer(n);
	finalize_init();
}

Network::~Network() { delete cost_function; delete activation_function; }

static Network* network; // Het neurale netwerk die we gaan gebruiken.

void Network::FeedForward(MatrixXf& data, int x)
{
	int layers = layerNeurons.size();
	a[0] = data.row(x);
	for (int l = 1; l < layers; l++)
	{
		z[l] = w[l - 1] * a[l - 1] + b[l - 1]; // Feed-forward voor elke laag.
		if (l == layers - 1)
			a[l] = L_activation_function->Evaluate(z[l]);
		else
			a[l] = activation_function->Evaluate(z[l]);
	}
}

int Network::Classify(MatrixXf& data, int x)
{
	FeedForward(data, x);
	int layers = layerNeurons.size();
	int max_index = 0;
	float max = 0;
	for (int i = 0; i < a[layers - 1].size(); i++)
	{
		if (a[layers - 1](i) > max) {
			max = a[layers - 1](i);
			max_index = i;
		}
	}
	return max_index;
}

void Network::Train(int batch_size, int num_epochs, float learning_rate)
{
	Train(batch_size, num_epochs, learning_rate, 0);
}

void Network::Train(int batch_size, int num_epochs, float learning_rate, float lambda)
{
	std::cout << "[Network] Network has started training\n";
	int layers = layerNeurons.size();

	srand(time(NULL));
	Eigen::PermutationMatrix<Dynamic, Dynamic> p(tr_images.rows());
	p.setIdentity();
	int epoch = 0;
	while(epoch < num_epochs && !stop_training)
	{
		// De dataset moet gepermuteerd worden voor stochastic gradient descent,
		// hiervoor maken we een willekeurige permutatiematrix p om de datamatrix mee te vermenigvuldigen.
		std::cout << "[Network|Training] Shuffling data matrix..\n";
		std::random_shuffle(p.indices().data(), p.indices().data() + p.indices().size());
		tr_images = p * tr_images;
		tr_labels = p * tr_labels; // Vermenigvuldig ook de labels met p.
		std::cout << "[Network|Training] Data matrix has been shuffled\n";

		// Doe Stochastic Gradient Descent
		for (int set_index = 0; set_index + batch_size < tr_images.rows(); set_index += batch_size)
		{
			if(set_index % 1000 == 0) error = 0;
			for (int batch_index = 0; batch_index < batch_size; batch_index++)
			{
				// Feed-forward
				FeedForward(tr_images, batch_index + set_index);
				VectorXf y = GetExpectedLabel(set_index + batch_index);

				// Backpropagation
				for (int l = layers - 1; l > 0; l--)
				{
					VectorXf derivatives = activation_function->EvaluateDerivative(z[l]);
					if (l == layers - 1)
						deltas[l - 1] = cost_function->GetDeltaL(a[layers - 1], y, derivatives);
					else
						deltas[l - 1] = (w[l].transpose() * deltas[l]).cwiseProduct(derivatives);
					gradient_b[l - 1] += deltas[l - 1];
					gradient_w[l - 1] += deltas[l - 1] * a[l - 1].transpose();
				}
				
				// Bereken de fout van deze afbeeldingsvector, tel het op bij de mean-squared error.
				if (set_index % 1000 == 0)
				{
					error += cost_function->GetCost(a[layers - 1], y);
					if (do_regularisation)
					{
						error += reg_term->Evaluate(lambda, tr_images.size(), w);
					}
				}
			}

			// Update de waarden met de gradiënt
			for (int l = 1; l < layers; l++)
			{
				if (do_regularisation)
				{
					w[l - 1] -= learning_rate * (reg_term->EvaluateDerivative(lambda, tr_images.rows(), w[l - 1])
						+ (gradient_w[l - 1] / batch_size));
				}
				else
				{
					w[l - 1] -= (learning_rate / batch_size) * gradient_w[l - 1];
				}
				b[l - 1] -= (learning_rate / batch_size) * gradient_b[l - 1];
				gradient_w[l - 1].fill(0);
				gradient_b[l - 1].fill(0);
			}

			if (set_index % 1000 == 0)
			{
				error /= batch_size;
				std::cout << "[Network|Training] Error: " << error << "\n";
			}
		}

		// Check progress
		int size = ts_images.rows();
		float recognised = 0;
		for (int i = 0; i < size; i++)
			recognised += (Classify(ts_images, i) == ts_labels[i]);
		std::cout << "[Network|Training] Image recognition rate: " << recognised << " / " << size << "\n\n";
		std::cout << "[Network|Training] Epochs trained: " << ++epoch << "\n";
	}
	std::cout << "[Network] Final error is:" << error << "\n";
	std::cout << "[Network] Network has successfully been trained\n";
}

// Laad het neurale netwerk; hierin worden alle parameters van het netwerk
// vastgesteld. Hoeveelheid lagen, hoeveelheid neuronen in elke laag,
// activatiefunctie, et cetera.
void init_network()
{
	std::cout << "[Network|Init] Initializing neural network..\n";

	network = new Network({ 784, 64, 10 }, crossentropy_cost, sigmoid_activation);
	network->SetRegularisation(L2_regularisation);

	std::cout << "[Network|Init] Successfully initialized neural network\n";
}

int main(int argc, char **argv)
{
	state = window.loading_state;
	visual::DataGrid grid(window, &ts_images, pixelSixe, centerX, centerY); 

	while (!window.isClosed())
	{
		if (state == window.explore_state)
		{
			// TODO: doe dit op een competente manier
			grid.draw();
			int n = network->Classify(ts_images, grid.dataMatrix_row);
			//std::cout << "Het netwerk denkt dat dit een " << n << " is.\n";
			grid.pollEvents();
		}

		window.pollEvents();
		window.clear();

		if (state == window.loading_state)
		{
			init_mnist();
			init_network();
			grid.init_data();
			
			//network->Train(10, 40, 0.3350);
			//network->Train(10, 40, 0.3355);
			//network->Train(8, 40, 0.33480);
			//network->Train(8, 10, 3); // 95% met 1 32-neuron laag en kwadratische kostenfunctie.
			//network->Train(10, 5, 0.001); // shitty ReLU
			network->Train(10, 15, 0.8, 5.0);

			state = window.explore_state;
		}
	}
	return 0;
}