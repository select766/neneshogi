from typing import List, Tuple


class Color:
    """
    手番定数
    """
    BLACK = 0  # 先手
    WHITE = 1  # 後手
    COLOR_NB = 2

    @staticmethod
    def invert(color: int) -> int:
        """
        手番の反転
        :param color:
        :return:
        """
        return 1 - color


class Piece:
    """
    駒定数
    """

    # 駒種
    NO_PIECE = 0
    PAWN = 1  # 歩
    LANCE = 2  # 香
    KNIGHT = 3  # 桂
    SILVER = 4  # 銀
    BISHOP = 5  # 角
    ROOK = 6  # 飛
    GOLD = 7  # 金
    KING = 8  # 玉
    PRO_PAWN = 9  # と
    PRO_LANCE = 10  # 成香
    PRO_KNIGHT = 11  # 成桂
    PRO_SILVER = 12  # 成銀
    HORSE = 13  # 馬
    DRAGON = 14  # 竜
    QUEEN = 15  # 未使用

    # 先手の駒
    B_PAWN = 1
    B_LANCE = 2
    B_KNIGHT = 3
    B_SILVER = 4
    B_BISHOP = 5
    B_ROOK = 6
    B_GOLD = 7
    B_KING = 8
    B_PRO_PAWN = 9
    B_PRO_LANCE = 10
    B_PRO_KNIGHT = 11
    B_PRO_SILVER = 12
    B_HORSE = 13
    B_DRAGON = 14
    B_QUEEN = 15  # 未使用

    # 後手の駒
    W_PAWN = 17
    W_LANCE = 18
    W_KNIGHT = 19
    W_SILVER = 20
    W_BISHOP = 21
    W_ROOK = 22
    W_GOLD = 23
    W_KING = 24
    W_PRO_PAWN = 25
    W_PRO_LANCE = 26
    W_PRO_KNIGHT = 27
    W_PRO_SILVER = 28
    W_HORSE = 29
    W_DRAGON = 30
    W_QUEEN = 31  # 未使用

    PIECE_NB = 32
    PIECE_ZERO = 0
    PIECE_PROMOTE = 8
    PIECE_WHITE = 16
    PIECE_RAW_NB = 8
    PIECE_HAND_ZERO = PAWN  # 手駒の駒種最小値
    PIECE_HAND_NB = KING  # 手駒の駒種最大値+1

    PIECE_FROM_CHAR_TABLE = {"P": B_PAWN, "L": B_LANCE, "N": B_KNIGHT, "S": B_SILVER,
                             "B": B_BISHOP, "R": B_ROOK, "G": B_GOLD, "K": B_KING,
                             "+P": B_PRO_PAWN, "+L": B_PRO_LANCE, "+N": B_PRO_KNIGHT, "+S": B_PRO_SILVER,
                             "+B": B_HORSE, "+R": B_DRAGON,
                             "p": W_PAWN, "l": W_LANCE, "n": W_KNIGHT, "s": W_SILVER,
                             "b": W_BISHOP, "r": W_ROOK, "g": W_GOLD, "k": W_KING,
                             "+p": W_PRO_PAWN, "+l": W_PRO_LANCE, "+n": W_PRO_KNIGHT, "+s": W_PRO_SILVER,
                             "+b": W_HORSE, "+r": W_DRAGON, }
    CHAR_FROM_PIECE_TABLE = {v: k for k, v in PIECE_FROM_CHAR_TABLE.items()}

    @staticmethod
    def piece_from_char(c: str) -> int:
        """
        駒の文字から定数を計算、先手が大文字、後手が小文字
        成り駒も考慮
        :param c:
        :return:
        """
        return Piece.PIECE_FROM_CHAR_TABLE[c]

    @staticmethod
    def char_from_piece(piece: int) -> str:
        """
        駒に対応する文字を返す、入力は駒種でも可(大文字が返る)
        成り駒も考慮し、"+"が先頭に付加される
        :param piece:
        :return:
        """
        return Piece.CHAR_FROM_PIECE_TABLE[piece]

    @staticmethod
    def raw_pt_from_piece(piece: int) -> int:
        """
        駒から、成る前の駒種を計算
        :param piece:
        :return:
        """
        raw_pt = piece % Piece.PIECE_RAW_NB
        if raw_pt == 0:
            raw_pt = Piece.KING
        return raw_pt

    @staticmethod
    def is_exist(piece: int) -> bool:
        """
        駒が存在するかどうか(空のマスでないか)を判定する
        :param piece:
        :return:
        """
        return piece > Piece.PIECE_ZERO

    @staticmethod
    def is_color(piece: int, color: int) -> bool:
        """
        駒が特定の色かどうか判定する
        :param piece:
        :param color:
        :return:
        """
        if piece == Piece.PIECE_ZERO:
            return False
        return piece // Piece.PIECE_WHITE == color


class Square:
    """
    マス定数
    筋*9+段
    1筋を0、1段を0に割り当てる
    """
    SQ_NB = 81

    @staticmethod
    def from_file_rank(file: int, rank: int) -> int:
        return file * 9 + rank

    @staticmethod
    def from_file_rank_if_valid(file: int, rank: int) -> Tuple[int, bool]:
        sq = file * 9 + rank
        valid = file >= 0 and file < 9 and rank >= 0 and rank < 9
        return sq, valid

    @staticmethod
    def file_of(sq: int) -> int:
        """
        筋を返す
        :param sq:
        :return:
        """
        return sq // 9

    @staticmethod
    def rank_of(sq: int) -> int:
        """
        段を返す
        :param sq:
        :return:
        """
        return sq % 9


from yaneuraou import Move


class UndoMoveInfo:
    # TODO remove it
    pass
