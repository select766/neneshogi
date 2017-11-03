#pragma once
class Piece
{
public:
	Piece();
	~Piece();
	static const int NO_PIECE = 0;
	static const int
		PAWN = 1,  // 歩
		LANCE = 2,  // 香
		KNIGHT = 3,  // 桂
		SILVER = 4,  // 銀
		BISHOP = 5,  // 角
		ROOK = 6,  // 飛
		GOLD = 7,  // 金
		KING = 8,  // 玉
		PRO_PAWN = 9,  // と
		PRO_LANCE = 10,  // 成香
		PRO_KNIGHT = 11,  // 成桂
		PRO_SILVER = 12,  // 成銀
		HORSE = 13,  // 馬
		DRAGON = 14,  // 竜
		QUEEN = 15,  // 未使用

		// 先手の駒
		B_PAWN = 1,
		B_LANCE = 2,
		B_KNIGHT = 3,
		B_SILVER = 4,
		B_BISHOP = 5,
		B_ROOK = 6,
		B_GOLD = 7,
		B_KING = 8,
		B_PRO_PAWN = 9,
		B_PRO_LANCE = 10,
		B_PRO_KNIGHT = 11,
		B_PRO_SILVER = 12,
		B_HORSE = 13,
		B_DRAGON = 14,
		B_QUEEN = 15,  // 未使用

		// 後手の駒
		W_PAWN = 17,
		W_LANCE = 18,
		W_KNIGHT = 19,
		W_SILVER = 20,
		W_BISHOP = 21,
		W_ROOK = 22,
		W_GOLD = 23,
		W_KING = 24,
		W_PRO_PAWN = 25,
		W_PRO_LANCE = 26,
		W_PRO_KNIGHT = 27,
		W_PRO_SILVER = 28,
		W_HORSE = 29,
		W_DRAGON = 30,
		W_QUEEN = 31,  // 未使用

		PIECE_NB = 32,
		PIECE_ZERO = 0,
		PIECE_PROMOTE = 8,
		PIECE_WHITE = 16,
		PIECE_RAW_NB = 8,
		PIECE_HAND_ZERO = PAWN,  // 手駒の駒種最小値
		PIECE_HAND_NB = KING;  // 手駒の駒種最大値 + 1

	// 駒が特定の色かどうか判定する
	static inline bool is_color(int piece, int color)
	{
		if (piece == Piece::PIECE_ZERO)
		{
			return false;
		}
		return piece / Piece::PIECE_WHITE == color;
	}

	// 駒が存在するかどうか(空のマスでないか)を判定する
	static inline bool is_exist(int piece)
	{
		return piece != Piece::PIECE_ZERO;
	}
};

