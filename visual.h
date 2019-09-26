#pragma once
#include <iostream>
#include <SDL.h>
#include <string>

namespace visual
{
	class Window
	{
	public:
		Window(const char* title, int h, int w);
		~Window();

		void pollEvents();
		inline bool isClosed() const;
		
	private:
		bool init();

		const char* title;
		int w, h;
		bool closed = false;

		SDL_Window* window = nullptr;
	};

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
		if (!init())
		{
			closed = true;
		}
	}

	Window::~Window()
	{
		SDL_DestroyWindow(window);
		SDL_Quit();
	}

	bool Window::init()
	{
		std::cout << "Initializing SDL..\n";
		if (SDL_Init(SDL_INIT_VIDEO) != 0)
		{
			std::cerr << "SDL failed to initialize!\n";
		}
		else
		{
			std::cout << "Successfully initialized SDL\n";
		}

		std::cout << "Creating SDL window..\n";
		window = SDL_CreateWindow(title, SDL_WINDOWPOS_CENTERED,
			SDL_WINDOWPOS_CENTERED, w, h, 0);

		if (window == nullptr)
		{
			std::cerr << "Failed to create SDL window";
			return 0;
		}
		std::cout << "Successfully initialized SDL window!\n";
		return 1;
	}
}