// neneshogi_cpp.cpp : DLL アプリケーション用にエクスポートされる関数を定義します。
//

#include "stdafx.h"
#include <pybind11/pybind11.h>
#include "Square.h"
#include "Move.h"

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
}
