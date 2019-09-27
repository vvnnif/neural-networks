#pragma once
#include <iostream>
#include <SDL.h>
#include <string>
#include "ala.h"

#define uint unsigned int

using namespace ala;

namespace visual
{
	class Window
	{
	public:
		Window(const char* title, int w, int h);
		~Window();

		void pollEvents();
		void clear() const;
		inline bool isClosed() const;
	private:
		bool init();

		const char* title;
		int w, h;
		bool closed = false;

		SDL_Window* window = nullptr;

		int gridOffset_y = 0;
		int gridOffset_x = 0;
		float pixelScaleFactor = 1;
		int pixelGridWidth = -1;

	protected:
		SDL_Renderer* renderer = nullptr;
	};

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

		std::cout << "Initializing SDL..\n";
		if (SDL_Init(SDL_INIT_VIDEO) != 0)
		{
			std::cerr << "SDL failed to initialize!\n";
		}
		else
		{
			std::cout << "Successfully initialized SDL\n";
		}

		//=================// Initializeer het venster

		std::cout << "Creating SDL window..\n";
		window = SDL_CreateWindow(title, SDL_WINDOWPOS_CENTERED,
			SDL_WINDOWPOS_CENTERED, w, h, 0);

		if (window == nullptr)
		{
			std::cerr << "Failed to create SDL window\n";
			return 0;
		}
		std::cout << "Successfully created SDL window\n";

		//=================// Initializeer de renderer

		renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_PRESENTVSYNC);
		if (renderer == nullptr)
		{
			std::cerr << "Failed to create SDL renderer\n";
			return 0;
		}
		else
		{
			std::cout << "Successfully created SDL renderer\n";
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
		uint8_t r, g, b;
	};

	class DataGrid
	{
	public:
		DataGrid(Matrix<uint8_t>* data,
			const int _pixelScaleFactor,
			const int _gridOffset_x, const int _gridOffset_y);
		void draw() const;
		void init_data(const Window& window);
		void update_data();
	private:
		Matrix<uint8_t>* data;
		int pixelScaleFactor;
		int gridOffset_x, gridOffset_y;
		int dataMatrix_row;
		std::vector<Pixel*> pixels;
	};

	void DataGrid::init_data(const Window& window)
	{
		int cols = data->GetCols();
		for (int i = 0; i < cols; i++)
		{
			int pixelX = gridOffset_x + ((i % 28) * pixelScaleFactor);
			int pixelY = gridOffset_y + (std::floor(i / 28) * pixelScaleFactor);
			uint c = (uint)data->at(dataMatrix_row, i);
			Pixel* p = new Pixel(window, pixelScaleFactor, pixelScaleFactor, pixelX, pixelY, c, c, c);
			pixels.push_back(p);
		}
	}

	void DataGrid::update_data()
	{

	}

	DataGrid::DataGrid(Matrix<uint8_t>* _data, const int _pixelScaleFactor,
		const int _gridOffset_x, const int _gridOffset_y)
		: data(_data), pixelScaleFactor(_pixelScaleFactor),
		gridOffset_x(_gridOffset_x), gridOffset_y(_gridOffset_y)
	{
		dataMatrix_row = ala::GetUniformRandom<int>(0, data->GetRows() - 1);
	}

	void DataGrid::draw() const
	{
		std::cout << "[Debug] Drawing pixel to screen..\n";
		for (Pixel* p : pixels)
		{
			p->draw();
		}
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