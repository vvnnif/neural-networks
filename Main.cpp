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

// Hier is de visualisatie-component van het project, alhoewel het niets
// toevoegt aan de theorie achter het profielwerkstuk kunt u het natuurlijk
// wel bekijken.
#include "visual.h"

#define uint unsigned int

using namespace Eigen;

static std::vector<uint> tr_labels, ts_labels; // Labels
static Eigen::MatrixXf tr_images, ts_images; // Afbeeldingen

static int state;

const int im_width = 28;

const int win_height = 600, win_width = 800;
const char* win_title = "Een interessante titel";

static visual::Window window = visual::Window(win_title, win_width, win_height);

const int pixelSixe = 16; // Grootte van de 'pixels' die getekend worden.
const int centerY = (win_height / 2) - ((pixelSixe * 28) / 2) - 64;
const int centerX = (win_width / 2) - ((pixelSixe * 28) / 2);

template<typename Derived>
void print_mnist_image(MatrixBase<Derived>& data, int im_size, int r);
void init_mnist();

double sigmoid(double x);
double sigmoidPrime(double x);
void init_network();

template<typename T>
Matrix<T, Dynamic, 1> VectorToEigen(std::vector<T>& vec)
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
	ts_labels = dataset.test_labels;
	tr_labels = dataset.training_labels;
	std::cout << "[MNIST] Successfully converted labels\n";
}

// Print een afbeelding uit de mnist-dataset in de console, 
// uitgedrukt in de waarden van elke pixel.
// \param r De hoeveelste afbeelding die gelezen moet worden.
// \param im_size De breedte/lengte van de afbeeldingen.
// \param data De afbeeldingsmatrix om uit te lezen.
template<typename Derived>
[[deprecated]] void print_mnist_image(MatrixBase<Derived>& data, int im_size, int r)
{
	int s = im_size * im_size;
	for (int i = 0; i < s; i++)
	{
		if (i % im_size == 0) std::cout << "\n";
		std::cout << (uint)data(r, i) << " ";
	}
}

// De sigmoid-activatiefunctie, beeldt R af op [0, 1].
// ( sigma(x) = 1 / (1 + e^-x) )
// \param x Het getal om af te beelden op [0, 1].
double sigmoid(double x)
{
	return 1 / (1 + std::exp(-x));
}

// De afgeleide van de sigmoid-functie.
double sigmoidPrime(double x)
{
	return sigmoid(x) * (1 - sigmoid(x));
}

typedef double(*function)(double); // Voor het gemak, scheelt schrijfwerk

// De implementatie van het neurale netwerk. 
class Network
{
public:
	Network();
	~Network();

	void SetActivationFunction(const function _activation_function);
	function GetActivationFunction() const;
	void SetActivationFunctionDerivative(const function _activation_function_derivative);
	function GetActivationFunctionDerivative() const;

	Network* AddLayer(int n_neurons);

	void finalize_init();

	void FeedForward(int row);
	void Train(int num_images, int epochs);

	std::vector<MatrixXf> w;
	std::vector<VectorXf> a;
	std::vector<VectorXf> b;
	std::vector<VectorXf> z;
	std::vector<VectorXf> deltas;
	std::vector<VectorXf> expectedResults;
private:
	function activation_function = nullptr; // Activatiefunctie van het netwerk.
	function activation_function_derivative = nullptr; // Zijn afgeleide.
	std::vector<int> layerNeurons; // Aantal neuronen per verborgen laag.
	float MSE = 0;
};

void Network::finalize_init()
{
	std::cout << "[Network|Init] Initializing network values..\n";
	for (size_t i = 0; i < layerNeurons.size(); i++)
	{
		VectorXf zero = VectorXf::Zero(layerNeurons[i]); 
		a.push_back(zero);

		if (i > 0)
		{
			VectorXf randomBiasVector = 50 * VectorXf::Random(layerNeurons[i]);
			b.push_back(randomBiasVector);

			MatrixXf randomWeightMatrix = MatrixXf::Random(layerNeurons[i], layerNeurons[i - 1]);
			w.push_back(randomWeightMatrix);
		}
		if (i > 0 && i < layerNeurons.size() - 1)
		{
			deltas.push_back(zero);
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
Network::Network(){}
Network::~Network(){}

// \param _activation_function De activatiefunctie van het neurale netwerk, e.g sigmoid, reLU, etc.
void Network::SetActivationFunction(const function _activation_function)
{
	std::cout << "[Network|Init] Setting activation function..\n";
	activation_function = _activation_function;
}

function Network::GetActivationFunction() const
{
	return activation_function;
}

// \param _activation_function_derivative De afgeleide van de activatiefunctie.
void Network::SetActivationFunctionDerivative(const function _activation_function_derivative)
{
	std::cout << "[Network|Init] Setting activation function derivative..\n";
	activation_function_derivative = _activation_function_derivative;
}

function Network::GetActivationFunctionDerivative() const
{
	return activation_function_derivative;
}

static Network* network; // Het neurale netwerk die we gaan gebruiken.

// Update het netwerk; bepaal voor een gegeven inputvector de output, afhankelijk van de
// weights en biases. 
void Network::FeedForward(int row)
{
	for (int n = 1; n < layerNeurons.size(); n++)
	{
		a[n] = w[n - 1] * a[n - 1] + b[n - 1]; // Feed-forward voor elke laag.
		for (int i = 1; i < layerNeurons[n]; i++) 
		{
			z[i - 1] = (a[n]); // Bepaal de 'onaangetastte' tussenwaarden z van elke laag.
			a[n](i) = activation_function(a[n](i)); // Bepaal de activaties met behulp van de activatiefunctie.
		}
	}

	// Backpropagation.
	//
	// Stap 1: Bereken de delta-waarden van de een-na-laatste laag.
	// Deze delta is gelijk aan de partiële afgeleide van de kostenfunctie met respect tot de
	// tussenwaarden van de een-na-laatste laag.
	// delta[i] = #een vector met als componenten de partiële afgeleiden van de kostenfunctie
	// met respect tot de activatie in de een-na-laatste laag# o #een vector met als componenten
	// de afgeleide van de activatiefunctie geëvalueerd op de tussenwaarden van de een-na-laatste-laag#
	for (int i = layerNeurons.size() - 2; i > 0; i--)
	{
		for (int j = 0; j < layerNeurons[i]; j++)
		{
			deltas[i](j) = (expectedResults[i](j) - a[i](j)) * activation_function_derivative(z[i](j));
		}
	}

	// Bereken de error van deze afbeeldingsvector, tel het op bij de mean-squared error.
	VectorXf toSquare = expectedResults[tr_labels[row]] - a[layerNeurons.size() - 1];
	for (int i = 0; i < toSquare.rows(); i++)
	{
		MSE += (toSquare[i] * toSquare[i]) / 2;
	}
}

void Network::Train(int num_images, int epochs)
{
	std::cout << "[Network] Network has started training\n";
	for (int i = 0; i < epochs; i++)
	{
		for (int j = 0; j < num_images; j++)
		{
			FeedForward(j);
		}
		MSE /= num_images;
		std::cout << "[Network|Training] Epochs trained: " << i + 1 << "\n";
		std::cout << "[Network|Training] Mean squared error: " << MSE << "\n";
		MSE = 0;
	}
	std::cout << "[Network] Network has successfully been trained\n";
}

// Laad het neurale netwerk; hierin worden alle parameters van het netwerk
// vastgesteld. Hoeveelheid lagen, hoeveelheid neuronen in elke laag,
// activatiefunctie, et cetera.
void init_network()
{
	std::cout << "[Network|Init] Initializing neural network..\n";

	//================// Netwerkinitializatie start

	network = new Network();

	network->SetActivationFunction(&sigmoid);
	network->SetActivationFunctionDerivative(&sigmoidPrime);

	network->AddLayer(784);
	network->AddLayer(16);
	network->AddLayer(16);
	network->AddLayer(10);

	network->finalize_init();

	//================// Netwerkinitializatie stop

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
			grid.draw();
			grid.pollEvents();
		}

		window.pollEvents();
		window.clear();

		if (state == window.loading_state)
		{
			init_mnist();
			init_network();
			grid.init_data();

			network->Train(10000, 2);

			state = window.explore_state;
		}
	}
	return 0;
}