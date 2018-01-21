#include "stdafx.h"
#include "Move.h"
#include "Color.h"
#include "Piece.h"
#include "Square.h"


Move::Move()
	: Move(0, 0, 0, false, false)
{
}

Move::Move(int move_from, int move_to, int move_dropped_piece, bool is_promote, bool is_drop)
	: _move_from(move_from), _move_to(move_to), _move_dropped_piece(move_dropped_piece), _is_promote(is_promote), _is_drop(is_drop)
{
}


Move::~Move()
{
}

Move Move::make_move(int move_from, int move_to, bool is_promote)
{
	return Move(move_from, move_to, 0, is_promote, false);
}

Move Move::make_move_drop(int move_dropped_piece, int move_to)
{
	return Move(0, move_to, move_dropped_piece, false, true);
}

Move Move::from_usi_string(std::string move_usi)
{
	int to_file = move_usi[2] - '1';
	int to_rank = move_usi[3] - 'a';
	int to_sq = Square::from_file_rank(to_file, to_rank);
	int from_file = move_usi[0] - '1';
	if (from_file > 8)
	{
		int drop_pt = 0;
		switch (move_usi[0])
		{
		case 'P':
			drop_pt = Piece::PAWN;
			break;
		case 'L':
			drop_pt = Piece::LANCE;
			break;
		case 'N':
			drop_pt = Piece::KNIGHT;
			break;
		case 'S':
			drop_pt = Piece::SILVER;
			break;
		case 'B':
			drop_pt = Piece::BISHOP;
			break;
		case 'R':
			drop_pt = Piece::ROOK;
			break;
		case 'G':
			drop_pt = Piece::GOLD;
			break;
		}

		return Move::make_move_drop(drop_pt, to_sq);
	}
	else
	{
		int from_rank = move_usi[1] - 'a';
		bool is_promote = move_usi.length() >= 5;
		return Move::make_move(Square::from_file_rank(from_file, from_rank), to_sq, is_promote);
	}
}

std::string Move::to_usi_string()
{
	char to_file_c = Square::file_of(_move_to) + '1';
	char to_rank_c = Square::rank_of(_move_to) + 'a';
	char str_base[6] = { 0, 0, to_file_c, to_rank_c, 0, 0 };
	if (_is_drop)
	{
		const char *drop_chars = " PLNSBRG";
		char drop_char = drop_chars[_move_dropped_piece];
		str_base[0] = drop_char;
		str_base[1] = '*';
	}
	else
	{
		char from_file_c = Square::file_of(_move_from) + '1';
		char from_rank_c = Square::rank_of(_move_from) + 'a';
		str_base[0] = from_file_c;
		str_base[1] = from_rank_c;
		if (_is_promote)
		{
			str_base[4] = '+';
		}
	}
	return std::string(str_base);
}

bool Move::py_eq(Move & other)
{
	return _move_from == other._move_from && _move_to == other._move_to &&
		_move_dropped_piece == other._move_dropped_piece &&
		_is_promote == other._is_promote && _is_drop == other._is_drop;
}

int Move::hash()
{
	return _move_to + (_move_from << 7) + (_move_dropped_piece << 7) +
		(_is_drop ? 16384 : 0) + (_is_promote ? 32768 : 0);
}
