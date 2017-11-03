// neneshogi_cpp.cpp : DLL アプリケーション用にエクスポートされる関数を定義します。
//

#include "stdafx.h"

namespace py = pybind11;

PYBIND11_MODULE(neneshogi_cpp, m) {
	m.attr("the_answer") = 42;
	py::object world = py::cast("World");
	m.attr("what") = world;
}
