// Mijn eigen geschreven library voor o.a lineaire algebra, 
// documentatie is geschreven in het Engels omdat het een hobbyproject was.
#include "ala.h"	
#include <fstream>

// Recht van het internet afgehaald, is wel zo handig.
// Moest alsnog een paar aanpassingen maken om het te laten werken.
#include "mnist/mnist_reader_less.hpp"
#include <string>

#define uint unsigned int

using namespace ala;

void print_mnist_image(Matrix<uint8_t> data, int im_size, int r);

// Print een afbeelding uit de mnist-dataset in de console, 
// uitgedrukt in de numerieke waarden van elke pixel.
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

int main()
{
	// Lees de MNIST-dataset en bewaar alle data in vectoren (labels) en
	// matrices (afbeeldingen). 
	auto dataset = mnist::read_dataset();
	Vector<uint8_t> tr_labels(dataset.training_labels);
	Matrix<uint8_t> tr_images(dataset.training_images);
	Vector<uint8_t> ts_labels(dataset.test_labels);
	Matrix<uint8_t> ts_images(dataset.test_images);
	
	print_mnist_image(ts_images, 28, 1);

	std::cin.get();
}