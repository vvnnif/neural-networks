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

double sigmoid(double x);
double sigmoidDerivative(double x);
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

// De sigmoid-activatiefunctie, beeldt R af op [0, 1].
// ( sigma(x) = 1 / (1 + e^-x) )
// \param x Het getal om af te beelden op [0, 1].
double sigmoid(double x)
{
	return 1 / (1 + std::exp(-x));
}

double reLU(double x)
{
	return (x > 0 ? x : 0);
}

double reLUDerivative(double x)
{
	return (x > 0 ? 1 : 0);
}

// De afgeleide van de sigmoid-functie.
double sigmoidDerivative(double x)
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
	Network* SetDebugMode(bool b);
	Network* SetAnalysisMode(bool b);
	Network* SetChangeTracking(bool b);
	void FeedForward(MatrixXf& data, int x);
	int Classify(MatrixXf& data, int x);

	Network* AddLayer(int n_neurons);

	void finalize_init();

	void Train(int batch_size, int epochs, float learning_rate);

	std::vector<MatrixXf> w;
	std::vector<VectorXf> a, b, z, deltas, expectedResults;

	std::vector<VectorXf> gradient_b;
	std::vector<MatrixXf> gradient_w;
private:
	function activation_function = nullptr; // Activatiefunctie van het netwerk.
	function activation_function_derivative = nullptr; // Zijn afgeleide.
	std::vector<int> layerNeurons; // Aantal neuronen per verborgen laag.
	float MSE = 0;
	bool debug = 0, analysis = 0, stop_training = 0;
};

Network* Network::SetDebugMode(bool b)
{
	this->debug = b;
	return this;
}

Network* Network::SetAnalysisMode(bool b)
{
	this->analysis = b;
	return this;
}

Network* Network::SetChangeTracking(bool b)
{
	this->analysis = b;
	return this;
}

void Network::finalize_init()
{
	std::cout << "[Network|Init] Initializing network values..\n";
	for (size_t i = 0; i < layerNeurons.size(); i++)
	{
		VectorXf zero = VectorXf::Zero(layerNeurons[i]); 
		a.push_back(zero);
		z.push_back(zero);

		if (i > 0)
		{
			//VectorXf bias = VectorXf::Constant(layerNeurons[i], 0.1);
			b.push_back(zero);
			deltas.push_back(zero);

			// Een techniek die bekend staat als "He-initialization".
			//MatrixXf randomWeightMatrix = std::sqrt(2 / layerNeurons[i - 1]) * MatrixXf::Random(layerNeurons[i], layerNeurons[i - 1]);
			MatrixXf randomWeightMatrix = MatrixXf::Random(layerNeurons[i], layerNeurons[i - 1]);
			w.push_back(randomWeightMatrix);

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

void Network::FeedForward(MatrixXf& data, int x)
{
	int layers = layerNeurons.size();
	a[0] = data.row(x);
	for (int l = 1; l < layers; l++)
	{
		z[l] = w[l - 1] * a[l - 1] + b[l - 1]; // Feed-forward voor elke laag.
		for (int i = 0; i < layerNeurons[l]; i++)
		{
			a[l](i) = activation_function(z[l](i)); // Bepaal de activaties met behulp van de activatiefunctie.
		}
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
			if(set_index % 1000 == 0) MSE = 0;
			for (int batch_index = 0; batch_index < batch_size; batch_index++)
			{
				// Feed-forward
				FeedForward(tr_images, batch_index + set_index);

				// Backpropagation
				for (int l = layers - 1; l > 0; l--)
				{
					VectorXf derivatives(layerNeurons[l]);
					for (int i = 0; i < layerNeurons[l]; i++)
						derivatives(i) = activation_function_derivative(z[l](i));
					if (l == layers - 1)
						deltas[l - 1] = (a[l] - expectedResults[tr_labels[batch_index + set_index]]).cwiseProduct(derivatives);
					else
						deltas[l - 1] = (w[l].transpose() * deltas[l]).cwiseProduct(derivatives);
					gradient_b[l - 1] += deltas[l - 1];
					gradient_w[l - 1] += deltas[l - 1] * a[l - 1].transpose();
				}

				// Bereken de fout van deze afbeeldingsvector, tel het op bij de mean-squared error.
				if (set_index % 1000 == 0)
				{
					VectorXf toSquare = a[layers - 1] - expectedResults[tr_labels[batch_index + set_index]];
					MSE += toSquare.dot(toSquare);
				}
			}

			// Update de waarden met de gradiënt
			for (int l = 1; l < layers; l++)
			{
				b[l - 1] -= (learning_rate / batch_size) * gradient_b[l - 1];
				gradient_b[l - 1].fill(0);
				w[l - 1] -= (learning_rate / batch_size) * gradient_w[l - 1];
				gradient_w[l - 1].fill(0);
			}

			if (set_index % 1000 == 0)
			{
				MSE /= batch_size;
				std::cout << "[Network|Training] Mean squared error: " << MSE << "\n";
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
	std::cout << "[Network] Final MSE is:" << MSE << "\n";
	std::cout << "[Network] Network has successfully been trained\n";
}

// Laad het neurale netwerk; hierin worden alle parameters van het netwerk
// vastgesteld. Hoeveelheid lagen, hoeveelheid neuronen in elke laag,
// activatiefunctie, et cetera.
void init_network()
{
	std::cout << "[Network|Init] Initializing neural network..\n";

	//================// Netwerkinitialisatie start

	network = new Network();

	network->SetActivationFunction(&sigmoid);
	network->SetActivationFunctionDerivative(&sigmoidDerivative);

	network->AddLayer(784)->AddLayer(32)->AddLayer(10);
	network->SetDebugMode(0)->SetAnalysisMode(0);

	network->finalize_init();

	// Verplaats later
	/*
	std::vector<std::vector<float>> s_weights;
	for (MatrixXf mat : network->w)
	{
		std::vector<float> v(mat.rows() * mat.cols());
		Map<MatrixXf> mat(v.data(), mat.rows(), mat.cols());
		s_weights.push_back(v);
	}
	*/

	//================// Netwerkinitialisatie stop

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
			std::cout << "Het netwerk denkt dat dit een " << n << " is.\n";
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
			network->Train(8, 5, 3); // 95% met 1 32-neuron laag 
			//network->Train(10, 5, 0.001); // shitty ReLU

			state = window.explore_state;
		}
	}
	return 0;
}