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
		static enum window_state { loading_state, menu_state, explore_state, classification_state, draw_state };
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
		SDL_SetRenderDrawColor(renderer, 20, 20, 20, 255);
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
		srand(time(NULL));
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
		void clear_data();
		void make_pixels(int x, int y, int r);
		void pollEvents();
		bool IsDrawing();
		int dataMatrix_row;
		std::vector<bool> classification_flags;
		VectorXf GetImageVector();
		void SetDistributionVector(VectorXf x);
	private:
		MatrixXf* data;
		VectorXf distr;
		const int pixelScaleFactor;
		const int gridOffset_x, gridOffset_y;
		int mouse_x = 0, mouse_y = 0;
		int border_size;
		int brush_size = 1;
		const int distr_w = (28 * pixelScaleFactor) / 2;
		const int distr_h = (28 * pixelScaleFactor) * 0.66f;
		const int distrOffset = pixelScaleFactor * 2;
		const int gap_size = 4;
		bool drawing = 0;
		bool mouse_hold = 0;
		std::vector<Pixel*> pixels;
	};

	void DataGrid::SetDistributionVector(VectorXf x)
	{
		this->distr = x;
	}

	bool DataGrid::IsDrawing()
	{
		return drawing;
	}

	VectorXf DataGrid::GetImageVector()
	{
		VectorXf out = VectorXf::Zero(784);
		for (int i = 0; i < 784; i++)
		{
			out(i) = (float) pixels[i]->r / 255.0f;
		}
		return out;
	}

	DataGrid::~DataGrid()
	{
		for (Pixel* p : pixels) delete p;
	}

	void DataGrid::pollEvents()
	{
		SDL_Event event;

		if (SDL_PollEvent(&event))
		{
			if (event.type == SDL_MOUSEBUTTONDOWN)
			{
				mouse_hold = 1;
				std::cout << "button down\n";
			}
			if (event.type == SDL_MOUSEBUTTONUP)
			{
				mouse_hold = 0;
				std::cout << "button up\n";
			}
			if (event.type == SDL_KEYDOWN)
			{
				switch (event.key.keysym.sym)
				{
				case SDLK_RIGHT:
					if (!drawing)
					{
						dataMatrix_row < (data->rows() - 1) ? dataMatrix_row++ : 0;
						load_data(dataMatrix_row);
					}
					break;
				case SDLK_LEFT:
					if (!drawing)
					{
						dataMatrix_row > 0 ? dataMatrix_row-- : 0;
						load_data(dataMatrix_row);
					}
					break;
				case SDLK_DOWN:
					if (!drawing)
					{
						dataMatrix_row = rand() % (data->rows() - 1);
						load_data(dataMatrix_row);
					}
					break;
				case SDLK_j:
					brush_size++;
					break;
				case SDLK_f:
					brush_size = (brush_size > 0 ? brush_size - 1 : 0);
					break;
				case SDLK_d:
					clear_data();
					drawing = !drawing;
					if (drawing == 0)
						load_data(dataMatrix_row);
					break;
				case SDLK_r:
					if (drawing)
					{
						clear_data();
					}
					break;
				}
			}

			if (mouse_hold && drawing)
			{
				make_pixels(event.button.x, event.button.y, brush_size);
			}
		}
	}

	void DataGrid::make_pixels(int x_0, int y_0, int r)
	{
		int pixelX_0 = std::floor((x_0 - gridOffset_x) / pixelScaleFactor);
		int pixelY_0 = std::floor((y_0 - gridOffset_y) / pixelScaleFactor);
		for (int u = -r; u <= r; u++)
		{
			for (int v = -r; v <= r; v++)
			{
				int x = x_0 + u * pixelScaleFactor;
				int y = y_0 + v * pixelScaleFactor;
				if (x > gridOffset_x && y > gridOffset_y &&
					x < gridOffset_x + 28 * pixelScaleFactor && y < gridOffset_y + 28 * pixelScaleFactor)
				{
					int pixelX = std::floor((x - gridOffset_x) / pixelScaleFactor);
					int pixelY = std::floor((y - gridOffset_y) / pixelScaleFactor);
					int xDist = pixelX_0 - pixelX;
					int yDist = pixelY_0 - pixelY;
					int dsqr = xDist * xDist + yDist * yDist;
					if (dsqr <= r * r)
					{
						int i = 28 * pixelY + pixelX;
						int c = min(255, 255 * brush_size / (dsqr + 1));
						if (c > pixels[i]->r)
						{
							pixels[i]->r = c;
							pixels[i]->g = c;
							pixels[i]->b = c;
						}
					}
				}
			}
		}
	}

	void DataGrid::clear_data()
	{
		for (int i = 0; i < pixels.size(); i++)
		{
			pixels[i]->r = 0;
			pixels[i]->g = 0;
			pixels[i]->b = 0;
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
		for (int i = 0; i < data->cols(); i++)
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
		border_size = 28 * pixelScaleFactor + 2;
		srand(time(NULL));
		dataMatrix_row = rand() % (data->rows() - 1);
	}

	void DataGrid::draw() const
	{
		// Getallen
		for (Pixel* p : pixels) p->draw();

		// Randen
		SDL_Rect borders;
		borders.h = border_size;
		borders.w = border_size;
		borders.x = gridOffset_x - 1;
		borders.y = gridOffset_y - 1;
		SDL_SetRenderDrawColor(renderer, 140, 140, 0, 255);
		SDL_RenderDrawRect(renderer, &borders);
		
		// Kansverdeling
		for (int i = 0; i < 10; i++)
		{
			SDL_Rect bar;
			SDL_Rect bar_border;
			int h = (distr_h - 8 * gap_size) / 10;
			int y_0 = gridOffset_y + ((28 * pixelScaleFactor - distr_h) / 2);
			bar.w = distr(i) * distr_w;
			bar.h = h;
			bar.x = gridOffset_x + 28 * pixelScaleFactor + distrOffset;
			bar.y = y_0 + i * gap_size + i * h;
			bar_border.w = distr_w + 2;
			bar_border.h = h + 2;
			bar_border.x = bar.x - 1;
			bar_border.y = bar.y - 1;
			SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
			SDL_RenderFillRect(renderer, &bar);
			SDL_SetRenderDrawColor(renderer, 140, 140, 0, 255);
			SDL_RenderDrawRect(renderer, &bar_border);
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