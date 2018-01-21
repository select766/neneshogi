#pragma once
class Piece
{
public:
	Piece();
	~Piece();
	static const int NO_PIECE = 0;
	static const int
		PAWN = 1,  // ��
		LANCE = 2,  // ��
		KNIGHT = 3,  // �j
		SILVER = 4,  // ��
		BISHOP = 5,  // �p
		ROOK = 6,  // ��
		GOLD = 7,  // ��
		KING = 8,  // ��
		PRO_PAWN = 9,  // ��
		PRO_LANCE = 10,  // ����
		PRO_KNIGHT = 11,  // ���j
		PRO_SILVER = 12,  // ����
		HORSE = 13,  // �n
		DRAGON = 14,  // ��
		QUEEN = 15,  // ���g�p

		// ���̋�
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
		B_QUEEN = 15,  // ���g�p

		// ���̋�
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
		W_QUEEN = 31,  // ���g�p

		PIECE_NB = 32,
		PIECE_ZERO = 0,
		PIECE_PROMOTE = 8,
		PIECE_WHITE = 16,
		PIECE_RAW_NB = 8,
		PIECE_HAND_ZERO = PAWN,  // ���̋��ŏ��l
		PIECE_HAND_NB = KING;  // ���̋��ő�l + 1

	// �����̐F���ǂ������肷��
	static inline bool is_color(int piece, int color)
	{
		if (piece == Piece::PIECE_ZERO)
		{
			return false;
		}
		return piece / Piece::PIECE_WHITE == color;
	}

	// ����݂��邩�ǂ���(��̃}�X�łȂ���)�𔻒肷��
	static inline bool is_exist(int piece)
	{
		return piece != Piece::PIECE_ZERO;
	}
};

