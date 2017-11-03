#pragma once
#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "Color.h"
#include "Piece.h"
#include "Square.h"
#include "Move.h"
#include "UndoMoveInfo.h"

namespace py = pybind11;

class Position
{
public:
	uint8_t _board[81];
	uint8_t _hand[2][7];
	int side_to_move;
	int game_ply;
	Position();
	~Position();
	py::array_t<uint8_t> get_board();
	void set_board(py::array_t<uint8_t> src);
	py::array_t<uint8_t> Position::get_hand();
	void set_hand(py::array_t<uint8_t> src);
	void set_hirate();
	UndoMoveInfo do_move(Move move);
	void undo_move(UndoMoveInfo undo_move_info);
	bool eq_board(Position& other);
};

