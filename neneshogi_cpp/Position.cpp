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
	// 内容のコピーが返る模様
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
	// 内容のコピーが返る模様
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
		// 駒打ち
		// 持ち駒を減らす
		int pt_hand = move._move_dropped_piece - Piece::PIECE_HAND_ZERO;
		hand_type = side_to_move * 7 + pt_hand;
		last_hand_value = _hand[side_to_move][pt_hand];
		_hand[side_to_move][pt_hand] = last_hand_value - 1;

		// 駒を置く
		int piece = move._move_dropped_piece;
		if (side_to_move == Color::WHITE)
		{
			piece += Piece::PIECE_WHITE;
		}
		to_sq = move._move_to;
		_board[to_sq] = piece;

		// 置く前は駒がなかったはず
		from_sq = to_sq;
		last_from_value = last_to_value = Piece::PIECE_ZERO;
	}
	else
	{
		// 駒の移動
		from_sq = move._move_from;
		to_sq = move._move_to;
		uint8_t captured_piece = _board[to_sq];
		if (captured_piece != Piece::PIECE_ZERO)
		{
			// 持ち駒を増やす
			// 駒種に変換
			int pt = captured_piece % Piece::PIECE_RAW_NB;
			if (pt == 0)
			{
				// KINGをとることはないはずだが
				pt = Piece::KING;
			}
			int pt_hand = pt - Piece::PIECE_HAND_ZERO;
			hand_type = side_to_move * 7 + pt_hand;
			last_hand_value = _hand[side_to_move][pt_hand];
			_hand[side_to_move][pt_hand] = last_hand_value + 1;
		}
		else
		{
			// 持ち駒は不変
			// 便宜上、_hand[0][0]の値を保存
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

// 本当はZobrist hashとかのほうがよさそうだが簡単に使える実装でごまかす
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
	//駒の配置・持ち駒・手番が一致するかどうか調べる。
	//手数・指し手の履歴は考慮しない。
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
	{ { 0, -1 } }, // 歩
	{ {} }, // 香
	{ { -1, -2 },{ 1, -2 } }, // 桂
	{ { -1, -1 },{ 0,-1 },{ 1,-1 },{ -1,1 },{ 1,1 } },//銀
	{},//角
	{},//飛
	{ { -1,-1 },{ 0,-1 },{ 1,-1 },{ -1,0 },{ 1,0 },{ 0,1 } },//金
	{ { -1,-1 },{ 0,-1 },{ 1,-1 },{ -1,0 },{ 1,0 },{ -1,1 },{ 0,1 },{ 1,1 } },//玉
	{ { -1,-1 },{ 0,-1 },{ 1,-1 },{ -1,0 },{ 1,0 },{ 0,1 } },//と
	{ { -1,-1 },{ 0,-1 },{ 1,-1 },{ -1,0 },{ 1,0 },{ 0,1 } },//成香
	{ { -1,-1 },{ 0,-1 },{ 1,-1 },{ -1,0 },{ 1,0 },{ 0,1 } },//成桂
	{ { -1,-1 },{ 0,-1 },{ 1,-1 },{ -1,0 },{ 1,0 },{ 0,1 } },//成銀
	{ { 0,-1 },{ -1,0 },{ 1,0 },{ 0,1 } },//馬
	{ { -1,-1 },{ 1,-1 },{ -1,1 },{ 1,1 } },//竜
};

static const int _SHORT_ATTACK_TABLE_LEN[15] = {
	0,1,0,2,5,0,0,6,8,
	6,6,6,6,4,4
};

static const int _MAX_NON_PROMOTE_RANK_TABLE[15] = {
	0,
	3,  // 歩(必ず成る)
	2,  // 香(2段目では必ず成る)
	2,  // 桂
	0,  // 銀
	3,  // 角(必ず成る)
	3,  // 飛(必ず成る)
	0,  // 金
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
	{},  // 歩
	{ { 0, -1 } },  // 香
	{},  // 桂
	{},  // 銀
	{ { -1, -1 },{ 1, -1 },{ -1, 1 },{ 1, 1 } },  // 角
	{ { 0, -1 },{ -1, 0 },{ 1, 0 },{ 0, 1 } },  // 飛
	{},  // 金
	{},  // 玉
	{},  // と
	{},  // 成香
	{},  // 成桂
	{},  // 成銀
	{ { -1, -1 },{ 1, -1 },{ -1, 1 },{ 1, 1 } },  // 馬
	{ { 0, -1 },{ -1, 0 },{ 1, 0 },{ 0, 1 } },  // 竜
};

static const int _LONG_ATTACK_TABLE_LEN[15] = {
	0,0,1,0,0,4,4,0,0,
	0,0,0,0,4,4
};

static const int _MAX_DROP_RANK_TABLE[8] = {
	0,1,1,2,0,0,0,0
};

/*
盤上の駒を動かす手をすべて生成する。
先手番を前提とする。
ただし、香車の2段目・歩・角・飛の不成りおよび行き場のない駒を生じる手は除く。
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
			// 短い利きの処理
			for (int short_attack_i = 0; short_attack_i < _SHORT_ATTACK_TABLE_LEN[from_piece]; short_attack_i++)
			{
				int x = _SHORT_ATTACK_TABLE[from_piece][short_attack_i][0];
				int y = _SHORT_ATTACK_TABLE[from_piece][short_attack_i][1];
				int to_file = from_file + x;
				int to_rank = from_rank + y;
				// 盤内確認
				int to_sq = Square::from_file_rank_if_valid(to_file, to_rank);
				if (to_sq < 0)
				{
					continue;
				}
				uint8_t to_piece = _board[to_sq];
				// 自分の駒があるところには進めない
				if (Piece::is_color(to_piece, Color::BLACK))
				{
					continue;
				}
				if (to_rank >= max_non_promote_rank)
				{
					// 行き場のない駒にはならない(&無意味な不成ではない)
					move_list.push_back(Move::make_move(from_sq, to_sq, false));
				}
				if (can_promote && (from_rank < 3 || to_rank < 3))
				{
					// 成れる駒で、成る条件を満たす
					move_list.push_back(Move::make_move(from_sq, to_sq, true));
				}
			}

			//長い利きの処理
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
						// 自分の駒があるところには進めない
						break;
					}
					if (to_rank >= max_non_promote_rank && from_rank >= max_non_promote_rank)
					{
						// 成って損がないのに成らない状況以外(角・飛)
						move_list.push_back(Move::make_move(from_sq, to_sq, false));
					}
					if (can_promote && (from_rank < 3 || to_rank < 3))
					{
						// 成れる駒で、成る条件を満たす
						move_list.push_back(Move::make_move(from_sq, to_sq, true));
					}
					if (Piece::is_exist(to_piece))
					{
						// 白駒があるので、これ以上進めない
						break;
					}

				}
			}
		}
	}
}

/*
駒を打つ手をすべて生成する。
先手番を前提とする。
ただし、二歩・行き場のない駒を生じる手は除く。
*/
void Position::_generate_move_drop(std::vector<Move>& move_list)
{

	// 二歩を避けるため、歩がすでにある筋を列挙
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
				// 駒のある場所には打てない
				continue;
			}

			for (uint8_t pt = Piece::PIECE_HAND_ZERO; pt < Piece::PIECE_HAND_NB; pt++)
			{
				if (_hand[0][pt - Piece::PIECE_HAND_ZERO] > 0)
				{
					if (pt == Piece::B_PAWN && pawn_files[to_file])
					{
						// 二歩
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
		// 王手放置チェック
		if (_in_check_black())
		{
			// 後手番になっているのに先手が王手をかけられている
			legal = false;
		}
		// 打ち歩詰めチェック
		if (legal && m._is_drop && m._move_dropped_piece == Piece::PAWN)
		{
			/*
			王手放置のときにチェックすると、玉を取る手が生成されてバグる
			現在の手番(後手)が詰んでいるとき、打ち歩詰め
			玉の頭に打った時だけ判定すればよい
			*/
			int white_king_check_pos = m._move_to - 1; // 1段目に打つ手は生成しないので、必ず盤内
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
		// 王手の時はすべての手
		return generate_move_list();
	}
	else
	{
		// 王手でないときは、last_moveと行先が同じ手
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

// 先手玉の左上、上、右上、…に存在すると、王手を構成する後手の駒(短い利き)。
static const int _CHECK_SHORT_ATTACK_PIECES[8][13] = {
	{ Piece::W_SILVER, Piece::W_BISHOP, Piece::W_GOLD, Piece::W_KING, Piece::W_PRO_PAWN, Piece::W_PRO_LANCE,
	Piece::W_PRO_KNIGHT, Piece::W_PRO_SILVER, Piece::W_HORSE, Piece::W_DRAGON, -1 },  // 左上
	{ Piece::W_PAWN, Piece::W_LANCE, Piece::W_SILVER, Piece::W_ROOK, Piece::W_GOLD, Piece::W_KING, Piece::W_PRO_PAWN,
	Piece::W_PRO_LANCE,
	Piece::W_PRO_KNIGHT, Piece::W_PRO_SILVER, Piece::W_HORSE, Piece::W_DRAGON,-1 },  // 上
	{ Piece::W_SILVER, Piece::W_BISHOP, Piece::W_GOLD, Piece::W_KING, Piece::W_PRO_PAWN, Piece::W_PRO_LANCE,
	Piece::W_PRO_KNIGHT, Piece::W_PRO_SILVER, Piece::W_HORSE, Piece::W_DRAGON,-1 },  // 右上
	{ Piece::W_ROOK, Piece::W_GOLD, Piece::W_KING, Piece::W_PRO_PAWN, Piece::W_PRO_LANCE,
	Piece::W_PRO_KNIGHT, Piece::W_PRO_SILVER, Piece::W_HORSE, Piece::W_DRAGON,-1 },  // 左
	{ Piece::W_ROOK, Piece::W_GOLD, Piece::W_KING, Piece::W_PRO_PAWN, Piece::W_PRO_LANCE,
	Piece::W_PRO_KNIGHT, Piece::W_PRO_SILVER, Piece::W_HORSE, Piece::W_DRAGON,-1 },  // 右
	{ Piece::W_SILVER, Piece::W_BISHOP, Piece::W_KING, Piece::W_HORSE, Piece::W_DRAGON,-1 },  // 左下
	{ Piece::W_ROOK, Piece::W_GOLD, Piece::W_KING, Piece::W_PRO_PAWN, Piece::W_PRO_LANCE,
	Piece::W_PRO_KNIGHT, Piece::W_PRO_SILVER, Piece::W_HORSE, Piece::W_DRAGON,-1 },  // 下
	{ Piece::W_SILVER, Piece::W_BISHOP, Piece::W_KING, Piece::W_HORSE, Piece::W_DRAGON,-1 },  // 右下
};

// 先手玉の左上、上、右上、…に存在すると、王手を構成する後手の駒(長い利き)。
static const int _CHECK_LONG_ATTACK_PIECES[8][4] = {
	{ Piece::W_BISHOP, Piece::W_HORSE,-1 },  // 左上
	{ Piece::W_LANCE, Piece::W_ROOK, Piece::W_DRAGON,-1 },  // 上
	{ Piece::W_BISHOP, Piece::W_HORSE,-1 },  // 右上
	{ Piece::W_ROOK, Piece::W_DRAGON,-1 },  // 左
	{ Piece::W_ROOK, Piece::W_DRAGON,-1 },  // 右
	{ Piece::W_BISHOP, Piece::W_HORSE,-1 },  // 左下
	{ Piece::W_ROOK, Piece::W_DRAGON,-1 },  // 下
	{ Piece::W_BISHOP, Piece::W_HORSE,-1 },  // 右下
};

/*
先手が王手された状態かどうかをチェックする。
先手が指して、後手番状態で呼び出すことも可能。この場合、王手放置のチェックとなる。
*/
bool Position::_in_check_black()
{
	/*
	先手玉からみて各方向に後手の駒があれば、王手されていることになる。
	例えば、先手玉の1つ上(y-方向)に後手歩があれば王手。
	先手玉の右下に、他の駒に遮られずに角があれば王手。
	長い利きの場合、途中のマスがすべて空でなければならない。
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
			// 隣に駒があるなら、それが玉に効く種類かどうか判定
			for (int pt_i = 0; pt_i < 13; pt_i++)
			{
				int pt_tmp = _CHECK_SHORT_ATTACK_PIECES[dir_i][pt_i];
				if (pt_tmp < 0)
				{
					break;
				}
				if (att_piece == pt_tmp)
				{
					// 短い利きが有効
					return true;
				}
			}
		}
		else
		{
			// マスが空なら、長い利きをチェック
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
						// 長い利きが有効
						return true;
					}
				}
				if (Piece::is_exist(att_piece))
				{
					// 空白以外の駒があるなら利きが切れる
					break;
				}
			}
		}
	}

	// 桂馬の利きチェック
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
			// 桂馬がいる
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
逆の手番から見た盤面に変化させる。
盤面・持ち駒・手番を反転。
*/
void Position::rotate_position_inplace()
{
	// 盤面を180度回し、駒の色を入れ替える。
	for (int sq = 0; sq < ((Square::SQ_NB + 1) / 2); sq++)
	{
		int inv_sq = Square::SQ_NB - 1 - sq;
		int sq_item = _board[sq];
		int inv_sq_item = _board[inv_sq];
		_board[sq] = _ROTATE_PIECE_TABLE[inv_sq_item];
		_board[inv_sq] = _ROTATE_PIECE_TABLE[sq_item];
	}

	// 持ち駒を入れ替える。
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

	// 盤上の駒
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

	// 持ち駒
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

	// 段・筋・定数1
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
