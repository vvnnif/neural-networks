#pragma once
#include <iostream>
#include <SDL.h>
#include <string>

#define uint unsigned int

namespace visual
{
	class Window
	{
	public:
		Window(const char* title, int h, int w);
		~Window();

		void pollEvents();
		void clear();
		void initPixelGrid(const int _im_size, const float _pixelScaleFactor,
			const int _gridOffset_x, const int _gridOffset_y);
		inline bool isClosed() const;
		
	private:
		bool init();

		const char* title;
		int w, h;
		bool closed = false;

		SDL_Window* window = nullptr;
		SDL_Renderer* renderer = nullptr;

		int gridOffset_y = 0;
		int gridOffset_x = 0;
		float pixelScaleFactor = 1;
		int pixelGridWidth = -1;

		const int pixelWidth = 1;
	};

	void Window::initPixelGrid(const int _im_size, const float _pixelScaleFactor, 
		const int _gridOffset_x, const int _gridOffset_y)
	{
		pixelScaleFactor = _pixelScaleFactor;
		pixelGridWidth = _im_size * pixelWidth * pixelScaleFactor;
		gridOffset_y = _gridOffset_y;
	}

	void Window::clear()
	{
		SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
		SDL_RenderClear(renderer);
		//=================// Teken hier
		
		SDL_Rect rect;
		rect.w = 16;
		rect.h = 16;
		rect.x = (w / 2) - (rect.w / 2);
		rect.y = (h / 2) - (rect.h / 2);
		
		SDL_SetRenderDrawColor(renderer, 170, 170, 170, 255);
		SDL_RenderFillRect(renderer, &rect);

		//=================//
		SDL_RenderPresent(renderer);
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

	Window::Window(const char* title, int h, int w)
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
}