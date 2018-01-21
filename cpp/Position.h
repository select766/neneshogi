#pragma once
#include <cstdint>
#include <vector>
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
	void copy_to(Position& other) const;
	int64_t hash() const;
	bool eq_board(Position& other);
	void _generate_move_move(std::vector<Move> &move_list);
	void _generate_move_drop(std::vector<Move> &move_list);
	std::vector<Move> generate_move_list();
	std::vector<Move> generate_move_list_nodrop();
	std::vector<Move> _generate_move_list_black(bool drop);
	// Ã~’Tõ—p
	std::vector<Move> generate_move_list_q(Move last_move);
	bool in_check();
	bool _in_check_black();
	void rotate_position_inplace();
	void make_dnn_input(int format, py::array_t<float, py::array::c_style | py::array::forcecast> dst);
	/* ‹l‚İ’Tõ
	‹l‚İ‚ªŒ©‚Â‚©‚Á‚½‚çA‰èè‡(std::vector<Move>)‚ğ•Ô‚·B
	’·‚³0‚È‚ç‹l‚İ‚ªŒ©‚Â‚©‚ç‚È‚©‚Á‚½ê‡(or©•ª‚ª‹l‚ñ‚Å‚¢‚éê‡)B
	ƒ^ƒvƒ‹‚Åè”‚Æè‡‚ğ•Ô‚µ‚½‚¢‚ªA‚â‚è•û‚ª‚í‚©‚ç‚È‚¢B
	*/
	std::vector<Move> mate_search();
};

