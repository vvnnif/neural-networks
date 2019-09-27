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

void print_mnist_image(Matrix<uint8_t> data, int im_size, int r);
void init_mnist();

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

int main(int argc, char **argv)
{
	init_mnist(); // Laad de MNIST-dataset.

	// Test, verwijder later.
	print_mnist_image(ts_images, im_width, 1);

	visual::DataGrid grid(&ts_images, 16, 0, 0);
	grid.init_data(window);

	while (!window.isClosed())
	{
		grid.draw();
		
		window.pollEvents();
		window.clear();
	}

	return 0;
}