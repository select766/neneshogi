#include "stdafx.h"
#include "Position.h"

static const uint8_t hirate_board[] =
{
	18, 0, 17, 0, 0, 0, 1, 0, 2,
	19, 21, 17, 0, 0, 0, 1, 6, 3,
	20, 0, 17, 0, 0, 0, 1, 0, 4,
	23, 0, 17, 0, 0, 0, 1, 0, 7,
	24, 0, 17, 0, 0, 0, 1, 0, 8,
	23, 0, 17, 0, 0, 0, 1, 0, 7,
	20, 0, 17, 0, 0, 0, 1, 0, 4,
	19, 22, 17, 0, 0, 0, 1, 5, 3,
	18, 0, 17, 0, 0, 0, 1, 0, 2
};

Position::Position()
{
}


Position::~Position()
{
}

py::array_t<uint8_t> Position::get_board()
{
	// ���e�̃R�s�[���Ԃ�͗l
	return py::array_t<uint8_t>(
		py::buffer_info(
			_board,
			sizeof(uint8_t),
			py::format_descriptor<uint8_t>::format(),
			1,
			{ 81 },
			{sizeof(uint8_t)}
		)
		);
}

void Position::set_board(py::array_t<uint8_t> src)
{
	auto info = src.request();
	memcpy(_board, info.ptr, 81);
}

py::array_t<uint8_t> Position::get_hand()
{
	// ���e�̃R�s�[���Ԃ�͗l
	return py::array_t<uint8_t>(
		py::buffer_info(
			_hand,
			sizeof(uint8_t),
			py::format_descriptor<uint8_t>::format(),
			2,
			{ 2, 7 },
			{ sizeof(uint8_t) * 7, sizeof(uint8_t) }
		)
		);
}

void Position::set_hand(py::array_t<uint8_t> src)
{
	auto info = src.request();
	memcpy(_hand, info.ptr, 14);
}

void Position::set_hirate()
{
	memcpy(_board, hirate_board, sizeof(hirate_board));
	memset(_hand, 0, sizeof(_hand));
	side_to_move = Color::BLACK;
	game_ply = 1;
}

UndoMoveInfo Position::do_move(Move move)
{
	int from_sq, to_sq, hand_type;
	uint8_t last_from_value, last_to_value, last_hand_value;
	if (move._is_drop)
	{
		// ��ł�
		// ����������炷
		int pt_hand = move._move_dropped_piece - Piece::PIECE_HAND_ZERO;
		hand_type = side_to_move * 7 + pt_hand;
		last_hand_value = _hand[side_to_move][pt_hand];
		_hand[side_to_move][pt_hand] = last_hand_value - 1;

		// ���u��
		int piece = move._move_dropped_piece;
		if (side_to_move == Color::WHITE)
		{
			piece += Piece::PIECE_WHITE;
		}
		to_sq = move._move_to;
		_board[to_sq] = piece;

		// �u���O�͋�Ȃ������͂�
		from_sq = to_sq;
		last_from_value = last_to_value = Piece::PIECE_ZERO;
	}
	else
	{
		// ��̈ړ�
		from_sq = move._move_from;
		to_sq = move._move_to;
		uint8_t captured_piece = _board[to_sq];
		if (captured_piece != Piece::PIECE_ZERO)
		{
			// ������𑝂₷
			// ���ɕϊ�
			int pt = captured_piece % Piece::PIECE_RAW_NB;
			if (pt == 0)
			{
				// KING���Ƃ邱�Ƃ͂Ȃ��͂�����
				pt = Piece::KING;
			}
			int pt_hand = pt - Piece::PIECE_HAND_ZERO;
			hand_type = side_to_move * 7 + pt_hand;
			last_hand_value = _hand[side_to_move][pt_hand];
			_hand[side_to_move][pt_hand] = last_hand_value + 1;
		}
		else
		{
			// ������͕s��
			// �֋X��A_hand[0][0]�̒l��ۑ�
			hand_type = 0;
			last_hand_value = _hand[0][0];
		}

		last_from_value = _board[from_sq];
		_board[from_sq] = 0;
		last_to_value = captured_piece;
		_board[to_sq] = last_from_value + (move._is_promote ? Piece::PIECE_PROMOTE : 0);
	}
	side_to_move = Color::invert(side_to_move);
	game_ply++;
	return UndoMoveInfo(from_sq, last_from_value, to_sq, last_to_value, hand_type, last_hand_value);
}

void Position::undo_move(UndoMoveInfo undo_move_info)
{
	game_ply--;
	side_to_move = Color::invert(side_to_move);
	_hand[0][undo_move_info._hand_type] = undo_move_info._hand_value;
	_board[undo_move_info._from_sq] = undo_move_info._from_value;
	_board[undo_move_info._to_sq] = undo_move_info._to_value;
}

bool Position::eq_board(Position & other)
{
	//��̔z�u�E������E��Ԃ���v���邩�ǂ������ׂ�B
	//�萔�E�w����̗����͍l�����Ȃ��B
	if (side_to_move != other.side_to_move)
	{
		return false;
	}
	if (memcmp(_board, other._board, sizeof(_board)) != 0)
	{
		return false;
	}
	if (memcmp(_hand, other._hand, sizeof(_hand)) != 0)
	{
		return false;
	}
	return true;
}
