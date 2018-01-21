#pragma once
class Square
{
public:
	Square();
	~Square();
	static const int SQ_NB = 81;
	static inline int Square::from_file_rank(int file, int rank)
	{
		return file * 9 + rank;
	}

	static inline int Square::from_file_rank_if_valid(int file, int rank)
	{
		if (file < 0 || file >= 9 || rank < 0 || rank >= 9) {
			return -1;
		}
		return file * 9 + rank;
	}

	static inline int Square::file_of(int sq)
	{
		return sq / 9;
	}

	static inline int Square::rank_of(int sq)
	{
		return sq % 9;
	}
};

