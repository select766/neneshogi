#pragma once
class Move
{
public:
	int _move_from, _move_to, _move_dropped_piece;
	bool _is_promote, _is_drop;
	Move();
	Move(int move_from, int move_to, int move_dropped_piece, bool is_promote, bool is_drop);
	~Move();

	static Move make_move(int move_from, int move_to, bool is_promote = false);
	static Move make_move_drop(int move_dropped_piece, int move_to);
	static Move from_usi_string(std::string move_usi);
	std::string to_usi_string();
	bool py_eq(Move &other);
	int hash();
};

