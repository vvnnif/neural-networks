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

// SDL-library voor datavisualisatie. 
#include <SDL.h>

#include <string>
#include <fstream>
#include <deque>

// Hier is de visualisatie-component van het project, alhoewel het niets
// toevoegt aan de theorie achter het profielwerkstuk kunt u het natuurlijk
// wel bekijken.
#include "visual.h"

#define uint unsigned int

using namespace ala;

static Vector<uint8_t> tr_labels, ts_labels; // Labels
static Matrix<uint8_t> tr_images, ts_images; // Afbeeldingen
const int im_width = 28;

const int win_height = 600, win_width = 800;
const char* win_title = "Een interessante titel";

static visual::Window window = visual::Window(win_title, win_width, win_height);

const int pixelSixe = 16; // Grootte van de 'pixels' die getekend worden.
const int centerY = (win_height / 2) - ((pixelSixe * 28) / 2) - 64;
const int centerX = (win_width / 2) - ((pixelSixe * 28) / 2);

void print_mnist_image(Matrix<uint8_t> data, int im_size, int r);
void init_mnist();

double sigmoid(double x);
double sigmoidPrime(double x);
void init_network();

// Lees de MNIST-dataset en bewaar alle data in vectoren (labels) en
// matrices (afbeeldingen).
void init_mnist()
{
	std::cout << "Parsing MNIST data..\n";

	auto dataset = mnist::read_dataset();

	tr_labels = dataset.training_labels;
	tr_images = dataset.training_images;
	ts_labels = dataset.test_labels;
	ts_images = dataset.test_images;

	std::cout << "Successfully parsed MNIST data\n";
}

// Print een afbeelding uit de mnist-dataset in de console, 
// uitgedrukt in de waarden van elke pixel.
// \param r De hoeveelste afbeelding die gelezen moet worden.
// \param im_size De breedte/lengte van de afbeeldingen.
// \param data De afbeeldingsmatrix om uit te lezen.
void print_mnist_image(Matrix<uint8_t> data, int im_size, int r)
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

class Network
{
public:
	Network(int input_neurons, int output_neurons);
	~Network();

	void SetActivationFunction(const function _activation_function);
	function GetActivationFunction() const;
	void SetActivationFunctionDerivative(const function _activation_function_derivative);
	function GetActivationFunctionDerivative() const;

	Network* AddHiddenLayer(int n_neurons);
private:
	function activation_function = nullptr; // Activatiefunctie van het netwerk.
	function activation_function_derivative = nullptr; // Zijn afgeleide.
	std::vector<int> hiddenLayerNeurons; // Aantal neuronen per verborgen laag.
	int inputNeurons, outputNeurons; // Aantal neuronen in de input- en outputlagen.
};

// Voeg een verborgen laag toe aan het netwerk.
// \param n_neurons Aantal neuronen van de verborgen laag.
Network* Network::AddHiddenLayer(int n_neurons)
{
	hiddenLayerNeurons.push_back(n_neurons);
	std::cout << "[Network] Adding hidden layer with " << n_neurons << " neurons..\n";
	return this;
}

// \param input_neurons Aantal neuronen in de inputlaag (aantal pixels in afbeelding)
// \param output_neurons Aantal neuronen in de outputlaag (aantal verschillende te herkennen getallen)
Network::Network(int input_neurons, int output_neurons)
	: inputNeurons(input_neurons), outputNeurons(output_neurons){}
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
	std::cout << "Initializing neural network..\n";

	//================// Netwerkinitializatie start

	network = new Network(tr_images.GetCols(), 10);

	network->SetActivationFunction(&sigmoid);
	network->SetActivationFunctionDerivative(&sigmoidPrime);
	network->AddHiddenLayer(16);
	network->AddHiddenLayer(8);

	//================// Netwerkinitializatie stop

	std::cout << "Successfully initialized neural network\n";
}

int main(int argc, char **argv)
{
	init_mnist(); // Laad de MNIST-dataset.
	init_network(); // Laad het neurale netwerk. 

	// Laad een cijfer uit de dataset voor later gebruik. 
	visual::DataGrid grid(window, &ts_images, pixelSixe, centerX, centerY); 
	grid.init_data();

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