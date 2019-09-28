//=======================================================================
// Finn van Vlaanderen
// A6A
// Profielwerkstuk "Neurale Netwerken"
// Programmeergedeelte / deelvraag 4
//=======================================================================

// Mijn eigen geschreven library voor o.a lineaire algebra, 
// documentatie is geschreven in het Engels omdat het een hobbyproject was.
#include "ala.h"	

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

using namespace ala;

static Vector<uint> tr_labels, ts_labels; // Labels
static Matrix<uint> tr_images, ts_images; // Afbeeldingen
const int im_width = 28;

const int win_height = 600, win_width = 800;
const char* win_title = "Een interessante titel";

static visual::Window window = visual::Window(win_title, win_width, win_height);

const int pixelSixe = 16; // Grootte van de 'pixels' die getekend worden.
const int centerY = (win_height / 2) - ((pixelSixe * 28) / 2) - 64;
const int centerX = (win_width / 2) - ((pixelSixe * 28) / 2);

void print_mnist_image(Matrix<float> data, int im_size, int r);
void init_mnist();

double sigmoid(double x);
double sigmoidPrime(double x);
void init_network();

// Lees de MNIST-dataset en bewaar alle data in vectoren (labels) en
// matrices (afbeeldingen).
void init_mnist()
{
	std::cout << "[MNIST] Parsing MNIST data..\n";

	auto dataset = mnist::read_dataset();

	tr_labels = dataset.training_labels;
	tr_images = dataset.training_images;
	ts_labels = dataset.test_labels;
	ts_images = dataset.test_images;
	
	std::cout << "[MNIST] Successfully parsed MNIST data\n";
}

// Print een afbeelding uit de mnist-dataset in de console, 
// uitgedrukt in de waarden van elke pixel.
// \param r De hoeveelste afbeelding die gelezen moet worden.
// \param im_size De breedte/lengte van de afbeeldingen.
// \param data De afbeeldingsmatrix om uit te lezen.
void print_mnist_image(Matrix<uint> data, int im_size, int r)
{
	int s = im_size * im_size;
	for (int i = 0; i < s; i++)
	{
		if (i % im_size == 0) std::cout << "\n";
		std::cout << (uint)data.at(r, i) << " ";
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

	void init_biases();
	void init_weights();
	void init_activations();

	void FeedForward(int row);
private:
	function activation_function = nullptr; // Activatiefunctie van het netwerk.
	function activation_function_derivative = nullptr; // Zijn afgeleide.
	std::vector<int> layerNeurons; // Aantal neuronen per verborgen laag.

	std::vector<ala::Vector<float>> activations; 
	std::vector<ala::Vector<float>> biases;
	std::vector<ala::Matrix<float>> weights;
};

void Network::init_activations()
{
	std::cout << "[Network] Initializing activation vectors..\n";
	for (size_t i = 0; i < layerNeurons.size(); i++)
	{
		Vector<float> zero(layerNeurons[i]);
		activations.push_back(zero);
	}
	std::cout << "[Network] Successfully initialized activation vectors\n";
}

void Network::init_biases()
{
	std::cout << "[Network] Initializing bias vectors..\n";
	for (size_t i = 1; i < layerNeurons.size(); i++)
	{
		Vector<float> randomBiasVector = GetRandomVector<float>(-50, 50, layerNeurons[i]);
		biases.push_back(randomBiasVector);
	}
	std::cout << "[Network] Successfully initialized bias vectors\n";
}

void Network::init_weights()
{
	std::cout << "[Network] Initializing weight matrices..\n";
	for (size_t i = 1; i < layerNeurons.size(); i++)
	{
		Matrix<float> randomWeightMatrix =
			GetRandomMatrix<float>(-1, 1, layerNeurons[i], layerNeurons[i - 1]);
		weights.push_back(randomWeightMatrix);
	}
	std::cout << "[Network] Successfully initialized weight matrices\n";
}

// Update het netwerk; bepaal voor een gegeven inputvector de output, afhankelijk van de
// weights en biases. 
void Network::FeedForward(int row)
{
	std::cout << "[Network] Feeding forward image at index " << row << "..\n";

	Vector<float> a_0(tr_images.GetCols());
	for (int i = 0; i < a_0.size(); i++)
	{
		a_0[i] = ((float)tr_images.at(row, i)) / 255;
	}
	activations[0] = a_0;

	for (int n = 1; n < layerNeurons.size(); n++)
	{
		function f = GetActivationFunction();
		Vector<float> a_n = (weights[n - 1] * activations[n - 1]) + biases[n - 1];
		for (int i = 0; i < a_n.size(); i++)
		{
			a_n[i] = f(a_n[i]);
		}
		activations[n] = a_n;
	}

	std::cout << "[Network] Successfully fed forward image at index " << row << "\n";
}

// Voeg een laag toe aan het netwerk.
// \param n_neurons Aantal neuronen van de verborgen laag.
Network* Network::AddLayer(int n_neurons)
{
	layerNeurons.push_back(n_neurons);
	std::cout << "[Network] Adding layer with " << n_neurons << " neurons..\n";
	return this;
}

// \param input_neurons Aantal neuronen in de inputlaag (aantal pixels in afbeelding)
// \param output_neurons Aantal neuronen in de outputlaag (aantal verschillende te herkennen getallen)
Network::Network(){}
Network::~Network(){}

// \param _activation_function De activatiefunctie van het neurale netwerk, e.g sigmoid, reLU, etc.
void Network::SetActivationFunction(const function _activation_function)
{
	std::cout << "[Network] Setting activation function..\n";
	activation_function = _activation_function;
}

function Network::GetActivationFunction() const
{
	return activation_function;
}

// \param _activation_function_derivative De afgeleide van de activatiefunctie.
void Network::SetActivationFunctionDerivative(const function _activation_function_derivative)
{
	std::cout << "[Network] Setting activation function derivative..\n";
	activation_function_derivative = _activation_function_derivative;
}

function Network::GetActivationFunctionDerivative() const
{
	return activation_function_derivative;
}

static Network* network; // Het neurale netwerk die we gaan gebruiken.

// Laad het neurale netwerk; hierin worden alle parameters van het netwerk
// vastgesteld. Hoeveelheid lagen, hoeveelheid neuronen in elke laag,
// activatiefunctie, et cetera.
void init_network()
{
	std::cout << "[Network] Initializing neural network..\n";

	//================// Netwerkinitializatie start

	network = new Network();

	network->SetActivationFunction(&sigmoid);
	network->SetActivationFunctionDerivative(&sigmoidPrime);

	network->AddLayer(tr_images.GetCols());
	network->AddLayer(16);
	network->AddLayer(16);
	network->AddLayer(10);

	network->init_activations();
	network->init_biases();
	network->init_weights();
	
	//================// Netwerkinitializatie stop

	std::cout << "[Network] Successfully initialized neural network\n";
}

int main(int argc, char **argv)
{
	init_mnist(); // Laad de MNIST-dataset.
	init_network(); // Laad het neurale netwerk. 
	
	// Laad een cijfer uit de dataset voor later gebruik. 
	visual::DataGrid grid(window, &ts_images, pixelSixe, centerX, centerY); 
	grid.init_data();
	
	// Test, verwijder later.
	
	for (size_t i = 0; i < tr_images.GetRows(); i++)
	{
		network->FeedForward(i);
	}

	// Gebruik de pijlentoetsen om naar andere cijfers in de dataset te gaan.
	// links: index 1 minder
	// rechts: index 1 meer
	// beneden: willekeurige index
	while (!window.isClosed())
	{
		grid.draw(); // Teken het cijfer op het scherm.
		grid.pollEvents();

		window.pollEvents();
		window.clear();
	}

	return 0;
}