#pragma once
#include <iostream>
#include <SDL.h>
#include <string>
#include <Eigen/Dense>

#define uint unsigned int

using namespace Eigen;

typedef Matrix<uint, Dynamic, Dynamic> MatrixXui;

namespace visual
{
	class Window
	{
	public:
		Window(const char* title, int w, int h);
		Window();
		~Window();

		void pollEvents();
		void clear() const;
		inline bool isClosed() const;
		static enum window_state { loading_state, menu_state, explore_state, classification_state };
	private:
		bool init();

		const char* title;
		int w, h;
		bool closed = false;

		SDL_Window* window = nullptr;
	protected:
		SDL_Renderer* renderer = nullptr;
	};

	Window::Window(){}

	void Window::clear() const
	{
		SDL_RenderPresent(renderer);
		SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
		SDL_RenderClear(renderer);
	}

	void Window::pollEvents()
	{
		SDL_Event event;
		if (SDL_PollEvent(&event))
		{
			switch (event.type)
			{
			case SDL_QUIT:
				closed = true;
				break;
			default:
				break;
			}
		}
	}

	inline bool Window::isClosed() const { return closed; }

	Window::Window(const char* title, int w, int h)
		: title(title), h(h), w(w)
	{
		closed = !init();
	}

	Window::~Window()
	{
		SDL_DestroyWindow(window);
		SDL_DestroyRenderer(renderer);

		SDL_Quit();
	}

	bool Window::init()
	{
		//=================// Initializeer SDL

		std::cout << "[Graphics] Initializing SDL..\n";
		if (SDL_Init(SDL_INIT_VIDEO) != 0)
		{
			std::cerr << "[Graphics] SDL failed to initialize!\n";
		}
		else
		{
			std::cout << "[Graphics] Successfully initialized SDL\n";
		}

		//=================// Initializeer het venster

		std::cout << "[Graphics] Creating SDL window..\n";
		window = SDL_CreateWindow(title, SDL_WINDOWPOS_CENTERED,
			SDL_WINDOWPOS_CENTERED, w, h, 0);

		if (window == nullptr)
		{
			std::cerr << "[Graphics] Failed to create SDL window\n";
			return 0;
		}
		std::cout << "[Graphics] Successfully created SDL window\n";

		//=================// Initializeer de renderer

		renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_PRESENTVSYNC);
		if (renderer == nullptr)
		{
			std::cerr << "[Graphics] Failed to create SDL renderer\n";
			return 0;
		}
		else
		{
			std::cout << "[Graphics] Successfully created SDL renderer\n";
		}

		return 1;
	}

	class Pixel : public Window
	{
	public:
		Pixel(const Window& _window, int _width, int _height, int _x, int _y, uint _r, uint _g, uint _b);
		void draw() const;
	private:
		int width, height;
		int x, y;
		uint r, g, b;
	};

	class DataGrid : public Window
	{
	public:
		DataGrid(const Window& window, MatrixXf* data,
			const int _pixelScaleFactor,
			const int _gridOffset_x, const int _gridOffset_y);
		~DataGrid();
		void draw() const;
		void init_data();
		void load_data(int row);
		void pollEvents();
	private:
		MatrixXf* data;
		int pixelScaleFactor;
		int gridOffset_x, gridOffset_y;
		int dataMatrix_row;
		std::vector<Pixel*> pixels;
	};

	DataGrid::~DataGrid()
	{
		for (Pixel* p : pixels) delete p;
	}

	void DataGrid::pollEvents()
	{
		SDL_Event event;

		if (SDL_PollEvent(&event))
		{
			if (event.type == SDL_KEYDOWN)
			{
				switch (event.key.keysym.sym)
				{
				case SDLK_RIGHT:
					dataMatrix_row < (data->rows() - 1) ? dataMatrix_row++ : 0;
					load_data(dataMatrix_row);
					break;
				case SDLK_LEFT:
					dataMatrix_row > 0 ? dataMatrix_row-- : 0;
					load_data(dataMatrix_row);
					break;
				case SDLK_DOWN:
					srand(time(NULL));
					dataMatrix_row = rand() % (data->rows() - 1);
					load_data(dataMatrix_row);
					break;
				}
			}
		}
	}

	void DataGrid::init_data()
	{
		int cols = data->cols();
		for (int i = 0; i < cols; i++)
		{
			int pixelX = gridOffset_x + ((i % 28) * pixelScaleFactor);
			int pixelY = gridOffset_y + (std::floor(i / 28) * pixelScaleFactor);
			uint c = (uint)((*data)(dataMatrix_row, i) * 255);
			Pixel* p = new Pixel(*this, pixelScaleFactor, pixelScaleFactor, pixelX, pixelY, c, c, c);
			pixels.push_back(p);
		}
	}
	
	void DataGrid::load_data(int row)
	{
		dataMatrix_row = row;
		int cols = data->cols();
		for (int i = 0; i < cols; i++)
		{
			int pixelX = gridOffset_x + ((i % 28) * pixelScaleFactor);
			int pixelY = gridOffset_y + (std::floor(i / 28) * pixelScaleFactor);
			uint c = (uint)((*data)(dataMatrix_row, i) * 255);
			pixels[i] = new Pixel(*this, pixelScaleFactor, pixelScaleFactor, pixelX, pixelY, c, c, c);
		}
	}

	DataGrid::DataGrid(const Window& window, MatrixXf* _data, const int _pixelScaleFactor,
		const int _gridOffset_x, const int _gridOffset_y)
		: Window(window), data(_data), pixelScaleFactor(_pixelScaleFactor),
		gridOffset_x(_gridOffset_x), gridOffset_y(_gridOffset_y)
	{
		srand(time(NULL));
		dataMatrix_row = rand() % (data->rows() - 1);
	}

	void DataGrid::draw() const
	{
		for (Pixel* p : pixels) p->draw();
	}

	Pixel::Pixel(const Window& _window, int _width, int _height, int _x, int _y, uint _r, uint _g, uint _b)
		: Window(_window), height(_height), width(_width), x(_x), y(_y),
		r(_r), g(_g), b(_b) {}

	void Pixel::draw() const
	{
		SDL_Rect rect;
		rect.h = height;
		rect.w = width;
		rect.x = x;
		rect.y = y;

		SDL_SetRenderDrawColor(renderer, r, g, b, 255);
		SDL_RenderFillRect(renderer, &rect);
	}
}