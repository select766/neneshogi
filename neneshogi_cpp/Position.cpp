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
			{ sizeof(uint8_t) }
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

void Position::copy_to(Position &other) const
{
	memcpy(&other, this, sizeof(Position)); //POD
}

// �{����Zobrist hash�Ƃ��̂ق����悳���������ȒP�Ɏg��������ł��܂���
// https://ja.wikipedia.org/wiki/%E5%B7%A1%E5%9B%9E%E5%86%97%E9%95%B7%E6%A4%9C%E6%9F%BB
static uint32_t crc_table[256];

class crc_initializer
{
public:
	crc_initializer()
	{
		for (uint32_t i = 0; i < 256; i++) {
			uint32_t c = i;
			for (int j = 0; j < 8; j++) {
				c = (c & 1) ? (0xEDB88320 ^ (c >> 1)) : (c >> 1);
			}
			crc_table[i] = c;
		}

	}
};

static crc_initializer _crc_init_dummy;

static uint32_t crc32(const uint8_t *buf, size_t len) {
	uint32_t c = 0xFFFFFFFF;
	for (size_t i = 0; i < len; i++) {
		c = crc_table[(c ^ buf[i]) & 0xFF] ^ (c >> 8);
	}
	return c ^ 0xFFFFFFFF;
}

int64_t Position::hash() const
{
	uint32_t upper = crc32(&_board[0], 41) ^ crc32(((uint8_t *)_hand), 7);
	uint32_t lower = crc32(&_board[41], 40) ^ crc32(((uint8_t *)_hand) + 7, 7) ^ side_to_move;
	return (int64_t)(((uint64_t)upper << 32) | lower);
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

static const int _SHORT_ATTACK_TABLE[15][8][2] =
{
	{},
	{ { 0, -1 } }, // ��
	{ {} }, // ��
	{ { -1, -2 },{ 1, -2 } }, // �j
	{ { -1, -1 },{ 0,-1 },{ 1,-1 },{ -1,1 },{ 1,1 } },//��
	{},//�p
	{},//��
	{ { -1,-1 },{ 0,-1 },{ 1,-1 },{ -1,0 },{ 1,0 },{ 0,1 } },//��
	{ { -1,-1 },{ 0,-1 },{ 1,-1 },{ -1,0 },{ 1,0 },{ -1,1 },{ 0,1 },{ 1,1 } },//��
	{ { -1,-1 },{ 0,-1 },{ 1,-1 },{ -1,0 },{ 1,0 },{ 0,1 } },//��
	{ { -1,-1 },{ 0,-1 },{ 1,-1 },{ -1,0 },{ 1,0 },{ 0,1 } },//����
	{ { -1,-1 },{ 0,-1 },{ 1,-1 },{ -1,0 },{ 1,0 },{ 0,1 } },//���j
	{ { -1,-1 },{ 0,-1 },{ 1,-1 },{ -1,0 },{ 1,0 },{ 0,1 } },//����
	{ { 0,-1 },{ -1,0 },{ 1,0 },{ 0,1 } },//�n
	{ { -1,-1 },{ 1,-1 },{ -1,1 },{ 1,1 } },//��
};

static const int _SHORT_ATTACK_TABLE_LEN[15] = {
	0,1,0,2,5,0,0,6,8,
	6,6,6,6,4,4
};

static const int _MAX_NON_PROMOTE_RANK_TABLE[15] = {
	0,
	3,  // ��(�K������)
	2,  // ��(2�i�ڂł͕K������)
	2,  // �j
	0,  // ��
	3,  // �p(�K������)
	3,  // ��(�K������)
	0,  // ��
	0,
	0,
	0,
	0,
	0,
	0,
	0,
};

static const int _LONG_ATTACK_TABLE[15][4][2] = {
	{},
	{},  // ��
	{ { 0, -1 } },  // ��
	{},  // �j
	{},  // ��
	{ { -1, -1 },{ 1, -1 },{ -1, 1 },{ 1, 1 } },  // �p
	{ { 0, -1 },{ -1, 0 },{ 1, 0 },{ 0, 1 } },  // ��
	{},  // ��
	{},  // ��
	{},  // ��
	{},  // ����
	{},  // ���j
	{},  // ����
	{ { -1, -1 },{ 1, -1 },{ -1, 1 },{ 1, 1 } },  // �n
	{ { 0, -1 },{ -1, 0 },{ 1, 0 },{ 0, 1 } },  // ��
};

static const int _LONG_ATTACK_TABLE_LEN[15] = {
	0,0,1,0,0,4,4,0,0,
	0,0,0,0,4,4
};

static const int _MAX_DROP_RANK_TABLE[8] = {
	0,1,1,2,0,0,0,0
};

/*
�Տ�̋�𓮂���������ׂĐ�������B
���Ԃ�O��Ƃ���B
�������A���Ԃ�2�i�ځE���E�p�E��̕s���肨��эs����̂Ȃ���𐶂����͏����B
*/
void Position::_generate_move_move(std::vector<Move>& move_list)
{
	for (int from_file = 0; from_file < 9; from_file++)
	{
		for (int from_rank = 0; from_rank < 9; from_rank++)
		{
			int from_sq = Square::from_file_rank(from_file, from_rank);
			uint8_t from_piece = _board[from_sq];
			if (!Piece::is_color(from_piece, Color::BLACK))
			{
				continue;
			}
			bool can_promote = from_piece <= Piece::B_ROOK;
			int max_non_promote_rank = _MAX_NON_PROMOTE_RANK_TABLE[from_piece];
			// �Z�������̏���
			for (int short_attack_i = 0; short_attack_i < _SHORT_ATTACK_TABLE_LEN[from_piece]; short_attack_i++)
			{
				int x = _SHORT_ATTACK_TABLE[from_piece][short_attack_i][0];
				int y = _SHORT_ATTACK_TABLE[from_piece][short_attack_i][1];
				int to_file = from_file + x;
				int to_rank = from_rank + y;
				// �Փ��m�F
				int to_sq = Square::from_file_rank_if_valid(to_file, to_rank);
				if (to_sq < 0)
				{
					continue;
				}
				uint8_t to_piece = _board[to_sq];
				// �����̋����Ƃ���ɂ͐i�߂Ȃ�
				if (Piece::is_color(to_piece, Color::BLACK))
				{
					continue;
				}
				if (to_rank >= max_non_promote_rank)
				{
					// �s����̂Ȃ���ɂ͂Ȃ�Ȃ�(&���Ӗ��ȕs���ł͂Ȃ�)
					move_list.push_back(Move::make_move(from_sq, to_sq, false));
				}
				if (can_promote && (from_rank < 3 || to_rank < 3))
				{
					// ������ŁA��������𖞂���
					move_list.push_back(Move::make_move(from_sq, to_sq, true));
				}
			}

			//���������̏���
			for (int long_attack_i = 0; long_attack_i < _LONG_ATTACK_TABLE_LEN[from_piece]; long_attack_i++)
			{
				int x = _LONG_ATTACK_TABLE[from_piece][long_attack_i][0];
				int y = _LONG_ATTACK_TABLE[from_piece][long_attack_i][1];
				int to_file = from_file;
				int to_rank = from_rank;
				while (true)
				{
					to_file += x;
					to_rank += y;
					int to_sq = Square::from_file_rank_if_valid(to_file, to_rank);
					if (to_sq < 0)
					{
						break;
					}
					uint8_t to_piece = _board[to_sq];
					if (Piece::is_color(to_piece, Color::BLACK))
					{
						// �����̋����Ƃ���ɂ͐i�߂Ȃ�
						break;
					}
					if (to_rank >= max_non_promote_rank && from_rank >= max_non_promote_rank)
					{
						// �����đ����Ȃ��̂ɐ���Ȃ��󋵈ȊO(�p�E��)
						move_list.push_back(Move::make_move(from_sq, to_sq, false));
					}
					if (can_promote && (from_rank < 3 || to_rank < 3))
					{
						// ������ŁA��������𖞂���
						move_list.push_back(Move::make_move(from_sq, to_sq, true));
					}
					if (Piece::is_exist(to_piece))
					{
						// �������̂ŁA����ȏ�i�߂Ȃ�
						break;
					}

				}
			}
		}
	}
}

/*
���ł�����ׂĐ�������B
���Ԃ�O��Ƃ���B
�������A����E�s����̂Ȃ���𐶂����͏����B
*/
void Position::_generate_move_drop(std::vector<Move>& move_list)
{

	// ���������邽�߁A�������łɂ���؂��
	bool pawn_files[9];
	for (int to_file = 0; to_file < 9; to_file++)
	{
		pawn_files[to_file] = false;
		for (int to_rank = 0; to_rank < 9; to_rank++)
		{
			int to_sq = Square::from_file_rank(to_file, to_rank);
			uint8_t to_piece = _board[to_sq];
			if (to_piece == Piece::B_PAWN)
			{
				pawn_files[to_file] = true;
				break;
			}
		}
	}

	for (int to_file = 0; to_file < 9; to_file++)
	{
		for (int to_rank = 0; to_rank < 9; to_rank++)
		{
			int to_sq = Square::from_file_rank(to_file, to_rank);
			uint8_t to_piece = _board[to_sq];
			if (Piece::is_exist(to_piece))
			{
				// ��̂���ꏊ�ɂ͑łĂȂ�
				continue;
			}

			for (uint8_t pt = Piece::PIECE_HAND_ZERO; pt < Piece::PIECE_HAND_NB; pt++)
			{
				if (_hand[0][pt - Piece::PIECE_HAND_ZERO] > 0)
				{
					if (pt == Piece::B_PAWN && pawn_files[to_file])
					{
						// ���
						continue;
					}

					int max_drop_rank = _MAX_DROP_RANK_TABLE[pt];
					if (to_rank < max_drop_rank)
					{
						continue;
					}

					move_list.push_back(Move::make_move_drop(pt, to_sq));
				}
			}
		}
	}
}

std::vector<Move> Position::generate_move_list()
{
	if (side_to_move == Color::BLACK)
	{
		return _generate_move_list_black(true);
	}
	else
	{
		rotate_position_inplace();
		std::vector<Move> move_list;
		for (Move &rot_move : _generate_move_list_black(true))
		{
			int to_sq = Square::SQ_NB - 1 - rot_move._move_to;
			if (rot_move._is_drop)
			{
				move_list.push_back(Move::make_move_drop(rot_move._move_dropped_piece, to_sq));
			}
			else
			{
				int from_sq = Square::SQ_NB - 1 - rot_move._move_from;
				move_list.push_back(Move::make_move(from_sq, to_sq, rot_move._is_promote));
			}
		}
		rotate_position_inplace();
		return move_list;
	}
}

std::vector<Move> Position::_generate_move_list_black(bool drop)
{
	std::vector<Move> possible_list;
	_generate_move_move(possible_list);
	if (drop)
	{
		_generate_move_drop(possible_list);
	}
	std::vector<Move> legal_list;
	for (Move &m : possible_list)
	{
		bool legal = true;
		auto undo_info = do_move(m);
		// ������u�`�F�b�N
		if (_in_check_black())
		{
			// ���ԂɂȂ��Ă���̂ɐ�肪������������Ă���
			legal = false;
		}
		// �ł����l�߃`�F�b�N
		if (legal && m._is_drop && m._move_dropped_piece == Piece::PAWN)
		{
			/*
			������u�̂Ƃ��Ƀ`�F�b�N����ƁA�ʂ����肪��������ăo�O��
			���݂̎��(���)���l��ł���Ƃ��A�ł����l��
			�ʂ̓��ɑł������������肷��΂悢
			*/
			int white_king_check_pos = m._move_to - 1; // 1�i�ڂɑł�͐������Ȃ��̂ŁA�K���Փ�
			if (_board[white_king_check_pos] == Piece::W_KING)
			{
				auto ml = generate_move_list();
				if (ml.empty())
				{
					legal = false;
				}
			}

		}
		undo_move(undo_info);
		if (legal)
		{
			legal_list.push_back(m);
		}
	}

	return legal_list;
}


std::vector<Move> Position::generate_move_list_nodrop()
{
	if (side_to_move == Color::BLACK)
	{
		return _generate_move_list_black(false);
	}
	else
	{
		rotate_position_inplace();
		std::vector<Move> move_list;
		for (Move &rot_move : _generate_move_list_black(false))
		{
			int to_sq = Square::SQ_NB - 1 - rot_move._move_to;
			if (rot_move._is_drop)
			{
				move_list.push_back(Move::make_move_drop(rot_move._move_dropped_piece, to_sq));
			}
			else
			{
				int from_sq = Square::SQ_NB - 1 - rot_move._move_from;
				move_list.push_back(Move::make_move(from_sq, to_sq, rot_move._is_promote));
			}
		}
		rotate_position_inplace();
		return move_list;
	}
}


std::vector<Move> Position::generate_move_list_q(Move last_move)
{
	if (in_check())
	{
		// ����̎��͂��ׂĂ̎�
		return generate_move_list();
	}
	else
	{
		// ����łȂ��Ƃ��́Alast_move�ƍs�悪������
		std::vector<Move> candidate_list = generate_move_list_nodrop();
		std::vector<Move> q_list;
		for (Move &m : candidate_list)
		{
			if (m._move_to == last_move._move_to)
			{
				q_list.push_back(m);
			}
		}

		return q_list;
	}
}

bool Position::in_check()
{
	if (side_to_move == Color::BLACK)
	{
		return _in_check_black();
	}
	else
	{
		rotate_position_inplace();
		bool ret = _in_check_black();
		rotate_position_inplace();
		return ret;
	}
}

static const int _CHECK_ATTACK_DIRS[8][2] = {
	{-1,-1},
	{0,-1},
	{1,-1},
	{-1,0},
	{1,0},
	{-1,1},
	{0,1},
	{1,1}
};

// ���ʂ̍���A��A�E��A�c�ɑ��݂���ƁA������\��������̋�(�Z������)�B
static const int _CHECK_SHORT_ATTACK_PIECES[8][13] = {
	{ Piece::W_SILVER, Piece::W_BISHOP, Piece::W_GOLD, Piece::W_KING, Piece::W_PRO_PAWN, Piece::W_PRO_LANCE,
	Piece::W_PRO_KNIGHT, Piece::W_PRO_SILVER, Piece::W_HORSE, Piece::W_DRAGON, -1 },  // ����
	{ Piece::W_PAWN, Piece::W_LANCE, Piece::W_SILVER, Piece::W_ROOK, Piece::W_GOLD, Piece::W_KING, Piece::W_PRO_PAWN,
	Piece::W_PRO_LANCE,
	Piece::W_PRO_KNIGHT, Piece::W_PRO_SILVER, Piece::W_HORSE, Piece::W_DRAGON,-1 },  // ��
	{ Piece::W_SILVER, Piece::W_BISHOP, Piece::W_GOLD, Piece::W_KING, Piece::W_PRO_PAWN, Piece::W_PRO_LANCE,
	Piece::W_PRO_KNIGHT, Piece::W_PRO_SILVER, Piece::W_HORSE, Piece::W_DRAGON,-1 },  // �E��
	{ Piece::W_ROOK, Piece::W_GOLD, Piece::W_KING, Piece::W_PRO_PAWN, Piece::W_PRO_LANCE,
	Piece::W_PRO_KNIGHT, Piece::W_PRO_SILVER, Piece::W_HORSE, Piece::W_DRAGON,-1 },  // ��
	{ Piece::W_ROOK, Piece::W_GOLD, Piece::W_KING, Piece::W_PRO_PAWN, Piece::W_PRO_LANCE,
	Piece::W_PRO_KNIGHT, Piece::W_PRO_SILVER, Piece::W_HORSE, Piece::W_DRAGON,-1 },  // �E
	{ Piece::W_SILVER, Piece::W_BISHOP, Piece::W_KING, Piece::W_HORSE, Piece::W_DRAGON,-1 },  // ����
	{ Piece::W_ROOK, Piece::W_GOLD, Piece::W_KING, Piece::W_PRO_PAWN, Piece::W_PRO_LANCE,
	Piece::W_PRO_KNIGHT, Piece::W_PRO_SILVER, Piece::W_HORSE, Piece::W_DRAGON,-1 },  // ��
	{ Piece::W_SILVER, Piece::W_BISHOP, Piece::W_KING, Piece::W_HORSE, Piece::W_DRAGON,-1 },  // �E��
};

// ���ʂ̍���A��A�E��A�c�ɑ��݂���ƁA������\��������̋�(��������)�B
static const int _CHECK_LONG_ATTACK_PIECES[8][4] = {
	{ Piece::W_BISHOP, Piece::W_HORSE,-1 },  // ����
	{ Piece::W_LANCE, Piece::W_ROOK, Piece::W_DRAGON,-1 },  // ��
	{ Piece::W_BISHOP, Piece::W_HORSE,-1 },  // �E��
	{ Piece::W_ROOK, Piece::W_DRAGON,-1 },  // ��
	{ Piece::W_ROOK, Piece::W_DRAGON,-1 },  // �E
	{ Piece::W_BISHOP, Piece::W_HORSE,-1 },  // ����
	{ Piece::W_ROOK, Piece::W_DRAGON,-1 },  // ��
	{ Piece::W_BISHOP, Piece::W_HORSE,-1 },  // �E��
};

/*
��肪���肳�ꂽ��Ԃ��ǂ������`�F�b�N����B
��肪�w���āA���ԏ�ԂŌĂяo�����Ƃ��\�B���̏ꍇ�A������u�̃`�F�b�N�ƂȂ�B
*/
bool Position::_in_check_black()
{
	/*
	���ʂ���݂Ċe�����Ɍ��̋����΁A���肳��Ă��邱�ƂɂȂ�B
	�Ⴆ�΁A���ʂ�1��(y-����)�Ɍ���������Ή���B
	���ʂ̉E���ɁA���̋�ɎՂ�ꂸ�Ɋp������Ή���B
	���������̏ꍇ�A�r���̃}�X�����ׂċ�łȂ���΂Ȃ�Ȃ��B
	*/

	int bk_sq = 0;
	for (int sq = 0; sq < Square::SQ_NB; sq++)
	{
		if (_board[sq] == Piece::B_KING)
		{
			bk_sq = sq;
			break;
		}
	}

	int bk_file = Square::file_of(bk_sq);
	int bk_rank = Square::rank_of(bk_sq);
	for (int dir_i = 0; dir_i < 8; dir_i++)
	{
		int x = _CHECK_ATTACK_DIRS[dir_i][0];
		int y = _CHECK_ATTACK_DIRS[dir_i][1];
		int att_file = bk_file + x;//attacker's file
		int att_rank = bk_rank + y;
		int att_sq = Square::from_file_rank_if_valid(att_file, att_rank);
		if (att_sq < 0)
		{
			continue;
		}

		uint8_t att_piece = _board[att_sq];
		if (Piece::is_exist(att_piece))
		{
			// �ׂɋ����Ȃ�A���ꂪ�ʂɌ�����ނ��ǂ�������
			for (int pt_i = 0; pt_i < 13; pt_i++)
			{
				int pt_tmp = _CHECK_SHORT_ATTACK_PIECES[dir_i][pt_i];
				if (pt_tmp < 0)
				{
					break;
				}
				if (att_piece == pt_tmp)
				{
					// �Z���������L��
					return true;
				}
			}
		}
		else
		{
			// �}�X����Ȃ�A�����������`�F�b�N
			while (true)
			{
				att_file += x;
				att_rank += y;
				att_sq = Square::from_file_rank_if_valid(att_file, att_rank);
				if (att_sq < 0)
				{
					break;
				}
				att_piece = _board[att_sq];
				for (int pt_i = 0; pt_i < 4; pt_i++)
				{
					int pt_tmp = _CHECK_LONG_ATTACK_PIECES[dir_i][pt_i];
					if (pt_tmp < 0)
					{
						break;
					}
					if (att_piece == pt_tmp)
					{
						// �����������L��
						return true;
					}
				}
				if (Piece::is_exist(att_piece))
				{
					// �󔒈ȊO�̋����Ȃ痘�����؂��
					break;
				}
			}
		}
	}

	// �j�n�̗����`�F�b�N
	for (int x = -1; x < 2; x += 2)//-1,1
	{
		int att_file = bk_file + x;
		int att_rank = bk_rank - 2;
		int att_sq = Square::from_file_rank_if_valid(att_file, att_rank);
		if (att_sq < 0)
		{
			continue;
		}

		uint8_t att_piece = _board[att_sq];
		if (att_piece == Piece::W_KNIGHT)
		{
			// �j�n������
			return true;
		}
	}
	return false;
}

static const int _ROTATE_PIECE_TABLE[] = {
	Piece::NO_PIECE, Piece::W_PAWN, Piece::W_LANCE, Piece::W_KNIGHT,
	Piece::W_SILVER, Piece::W_BISHOP, Piece::W_ROOK, Piece::W_GOLD,
	Piece::W_KING, Piece::W_PRO_PAWN, Piece::W_PRO_LANCE, Piece::W_PRO_KNIGHT,
	Piece::W_PRO_SILVER, Piece::W_HORSE, Piece::W_DRAGON, Piece::W_QUEEN,
	Piece::NO_PIECE, Piece::B_PAWN, Piece::B_LANCE, Piece::B_KNIGHT,
	Piece::B_SILVER, Piece::B_BISHOP, Piece::B_ROOK, Piece::B_GOLD,
	Piece::B_KING, Piece::B_PRO_PAWN, Piece::B_PRO_LANCE, Piece::B_PRO_KNIGHT,
	Piece::B_PRO_SILVER, Piece::B_HORSE, Piece::B_DRAGON, Piece::B_QUEEN
};

/*
�t�̎�Ԃ��猩���Ֆʂɕω�������B
�ՖʁE������E��Ԃ𔽓]�B
*/
void Position::rotate_position_inplace()
{
	// �Ֆʂ�180�x�񂵁A��̐F�����ւ���B
	for (int sq = 0; sq < ((Square::SQ_NB + 1) / 2); sq++)
	{
		int inv_sq = Square::SQ_NB - 1 - sq;
		int sq_item = _board[sq];
		int inv_sq_item = _board[inv_sq];
		_board[sq] = _ROTATE_PIECE_TABLE[inv_sq_item];
		_board[inv_sq] = _ROTATE_PIECE_TABLE[sq_item];
	}

	// ����������ւ���B
	for (int i = 0; i < 7; i++)
	{
		int bh = _hand[0][i];
		int wh = _hand[1][i];
		_hand[1][i] = bh;
		_hand[0][i] = wh;
	}

	side_to_move = Color::invert(side_to_move);
}

void Position::make_dnn_input(int format, py::array_t<float, py::array::c_style | py::array::forcecast> dst)
{
	auto info = dst.request();
	float* dst_ptr = static_cast<float*>(info.ptr);//(61,9,9)

	bool rotate = false;
	if (side_to_move == Color::WHITE)
	{
		rotate_position_inplace();
		rotate = true;
	}

	memset(dst_ptr, 0, sizeof(float) * 61 * Square::SQ_NB);

	// �Տ�̋�
	for (int sq = 0; sq < Square::SQ_NB; sq++)
	{
		uint8_t piece = _board[sq];
		if (piece > 0)
		{
			int ch;
			if (piece >= Piece::W_PAWN)
			{
				ch = piece - Piece::W_PAWN + 14;
			}
			else
			{
				ch = piece - Piece::B_PAWN;
			}

			dst_ptr[ch * Square::SQ_NB + sq] = 1.0;
		}
	}

	// ������
	for (int color = 0; color < Color::COLOR_NB; color++)
	{
		for (int i = 0; i < (Piece::PIECE_HAND_NB-Piece::PIECE_HAND_ZERO); i++)
		{
			int hand_count = _hand[color][i];
			int ch = color * 7 + 28 + i;
			for (int sq = 0; sq < Square::SQ_NB; sq++)
			{
				dst_ptr[ch * Square::SQ_NB + sq] = hand_count;
			}
		}
	}

	// �i�E�؁E�萔1
	for (int sq = 0; sq < Square::SQ_NB; sq++)
	{
		dst_ptr[(Square::rank_of(sq) + 42) * Square::SQ_NB + sq] = 1.0;
		dst_ptr[(Square::file_of(sq) + 51) * Square::SQ_NB + sq] = 1.0;
		dst_ptr[60 * Square::SQ_NB + sq] = 1.0;
	}

	if (rotate)
	{
		rotate_position_inplace();
	}
}
