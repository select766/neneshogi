// neneshogi_cpp.cpp : DLL アプリケーション用にエクスポートされる関数を定義します。
//

#include "stdafx.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Square.h"
#include "Move.h"
#include "UndoMoveInfo.h"
#include "Position.h"

namespace py = pybind11;

PYBIND11_MODULE(neneshogi_cpp, m) {
	py::class_<Square>(m, "Square")
		.def_readonly_static("SQ_NB", &Square::SQ_NB)
		.def_static("file_of", &Square::file_of)
		.def_static("rank_of", &Square::rank_of)
		.def_static("from_file_rank", &Square::from_file_rank)
		.def_static("from_file_rank_if_valid",
			[](int file, int rank) {
		int sq = Square::from_file_rank_if_valid(file, rank);
		return py::make_tuple(sq, sq >= 0);
	});
	py::class_<Move>(m, "Move")
		.def(py::init<int, int, int, bool, bool>())
		.def_readonly("move_from", &Move::_move_from)
		.def_readonly("move_to", &Move::_move_to)
		.def_readonly("move_dropped_piece", &Move::_move_dropped_piece)
		.def_readonly("is_promote", &Move::_is_promote)
		.def_readonly("is_drop", &Move::_is_drop)
		.def_static("make_move", &Move::make_move, "make_move",
			py::arg("move_from"), py::arg("move_to"),
			py::arg("is_promote") = false)
		.def_static("make_move_drop", &Move::make_move_drop)
		.def_static("from_usi_string", &Move::from_usi_string)
		.def("to_usi_string", &Move::to_usi_string)
		.def("__str__", &Move::to_usi_string)
		.def("__eq__", &Move::py_eq)
		.def("__hash__", &Move::hash)
		.def(py::pickle(
			[](const Move &m) {
				return py::make_tuple(m._move_from, m._move_to, m._move_dropped_piece, m._is_promote, m._is_drop);
			},
			[](py::tuple t) {
				return Move(t[0].cast<int>(), t[1].cast<int>(), t[2].cast<int>(), t[3].cast<bool>(), t[4].cast<bool>());
			}));
	py::class_<UndoMoveInfo>(m, "UndoMoveInfo")
		.def(py::init<int, uint8_t, int, uint8_t, int, uint8_t>())
		.def(py::pickle(
			[](const UndoMoveInfo &u) {
				return py::make_tuple(u._from_sq, u._from_value, u._to_sq, u._to_value, u._hand_type, u._hand_value);
			},
			[](py::tuple t) {
				return UndoMoveInfo(t[0].cast<int>(), t[1].cast<uint8_t>(), t[2].cast<int>(), t[3].cast<uint8_t>(), t[4].cast<int>(), t[5].cast<uint8_t>());
			}));
	py::class_<Position>(m, "Position")
		.def(py::init<>())
		.def_property_readonly("board", &Position::get_board)
		.def_property_readonly("hand", &Position::get_hand)
		.def("set_board", &Position::set_board)
		.def("set_hand", &Position::set_hand)
		.def_readwrite("side_to_move", &Position::side_to_move)
		.def_readwrite("game_ply", &Position::game_ply)
		.def("set_hirate", &Position::set_hirate)
		.def("do_move", &Position::do_move)
		.def("undo_move", &Position::undo_move)
		.def("copy_to", &Position::copy_to)
		.def("hash", &Position::hash)
		.def("eq_board", &Position::eq_board)
		.def("generate_move_list", &Position::generate_move_list)
		.def("generate_move_list_q", &Position::generate_move_list_q)
		.def("make_dnn_input", &Position::make_dnn_input)
		.def("in_check", &Position::in_check)
		.def("mate_search", &Position::mate_search)
		.def(py::pickle(
			[](const Position &p) {
				std::vector<uint8_t> data(sizeof(Position));
				memcpy(&data[0], &p, sizeof(Position));
				return py::make_tuple(data);
			},
			[](py::tuple t) {
				Position p;
				std::vector<uint8_t> data = t[0].cast<std::vector<uint8_t> >();
				memcpy(&p, &data[0], sizeof(Position));
				return p;
			}));
}
