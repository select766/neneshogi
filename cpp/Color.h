#pragma once
class Color
{
public:
	Color();
	~Color();
	static const int BLACK = 0, WHITE = 1, COLOR_NB = 2;

	static inline int invert(int color)
	{
		return 1 - color;
	}
};

