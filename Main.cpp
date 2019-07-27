#include <iostream>
#include <thing.h>
#include <matrix.h>

// That one project will be rewritten.
// Todo: Also make ALALib2
// Todo: Unit tests
int main()
{
	ala::Matrix<int> m = ala::Matrix<int>(2, 3);
	ala::Matrix<int> n = ala::Matrix<int>(7, 7);

	ala::Matrix<int> o = m + n;

	std::cout << m.GetDataTemp() << std::endl;
	std::cin.get();
}