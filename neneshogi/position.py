"""
盤面を表現するクラス
関数名等は、可能な限りやねうら王に準拠
"""
from typing import List, Tuple

import numpy as np
from .move import Color, Piece, Square, Move, UndoMoveInfo
#from . import position_acc
from .neneshogi_cpp import Position as CPosition

class Position(CPosition):
    """
    盤面を表現するクラス
    """
    board: np.ndarray  # 盤面(81要素,np.uint8)
    hand: np.ndarray  # 持ち駒((2,7)要素,np.uint8)
    side_to_move: int  # 手番
    game_ply: int  # 手数(開始局面で1)

    def __init__(self):
        super().__init__()

    def copy(self):
        dst = Position()
        self.copy_to(dst)
        return dst

    def set_sfen(self, sfen: str) -> None:
        """
        SFEN形式の局面をセットする。
        :param sfen:
        :return:
        """
        board_str, color_str, hand_str, ply_str = sfen.split()
        board_ranks = board_str.split("/")
        board = np.zeros((Square.SQ_NB,), dtype=np.uint8)
        for rank, rank_line in enumerate(board_ranks):
            file_from_left = 0
            is_promote = False
            for token in rank_line:
                if token == "+":
                    is_promote = True
                    continue
                if token.isnumeric():
                    # 数値の指す数だけ空のマス
                    for _ in range(int(token)):
                        sq = Square.from_file_rank(8 - file_from_left, rank)
                        board[sq] = Piece.NO_PIECE
                        file_from_left += 1
                else:
                    # 駒
                    piece = Piece.piece_from_char(token)
                    if is_promote:
                        piece += Piece.PIECE_PROMOTE
                    sq = Square.from_file_rank(8 - file_from_left, rank)
                    board[sq] = piece
                    file_from_left += 1
                is_promote = False
            assert file_from_left == 9
        self.set_board(board)
        assert np.all(self.board == board)

        hand = np.zeros((Color.COLOR_NB, (Piece.PIECE_HAND_NB - Piece.PIECE_HAND_ZERO)), dtype=np.uint8)
        if hand_str != "-":
            num_piece_str = ""
            for token in hand_str:
                if token.isnumeric():
                    # 駒の数が10以上のときがあるので注意
                    num_piece_str += token
                else:
                    piece = Piece.piece_from_char(token)
                    piece_color = Color.WHITE if Piece.is_color(piece, Color.WHITE) else Color.BLACK
                    num_piece = int(num_piece_str) if len(num_piece_str) > 0 else 1
                    hand[piece_color, Piece.raw_pt_from_piece(piece) - Piece.PIECE_HAND_ZERO] = num_piece
                    num_piece_str = ""
        self.set_hand(hand)
        assert np.all(self.hand == hand)

        self.side_to_move = Color.WHITE if color_str == "w" else Color.BLACK
        self.game_ply = int(ply_str)

    def get_sfen(self) -> str:
        """
        SFEN形式で局面を表現する
        :return:
        """

        # 盤面
        # SFENは段ごとに、左から右に走査する
        sfen = ""
        board = self.board
        for y in range(9):
            if y > 0:
                sfen += "/"
            blank_len = 0
            for x in range(9):
                sq = (8 - x) * 9 + y
                piece = board[sq]
                if piece != Piece.PIECE_ZERO:
                    if blank_len > 0:
                        sfen += str(blank_len)
                        blank_len = 0
                    sfen += Piece.char_from_piece(piece)
                else:
                    blank_len += 1
            if blank_len > 0:
                sfen += str(blank_len)

        # 手番
        if self.side_to_move == Color.BLACK:
            sfen += " b "
        else:
            sfen += " w "

        # 持ち駒
        # 同じ局面・手数の時にSFENを完全一致させるため、飛、角、金、銀、桂、香、歩の順とする
        hand_pieces = ""
        hand = self.hand
        for color in range(Color.COLOR_NB):
            hand_for_color = hand[color]
            piece_color_offset = color * Piece.PIECE_WHITE
            for pt in [Piece.ROOK, Piece.BISHOP, Piece.GOLD, Piece.SILVER, Piece.KNIGHT, Piece.LANCE, Piece.PAWN]:
                piece_ct = hand_for_color[pt - Piece.PIECE_HAND_ZERO]
                if piece_ct > 0:
                    if piece_ct > 1:
                        hand_pieces += str(piece_ct)
                    hand_pieces += Piece.char_from_piece(pt + piece_color_offset)
        if len(hand_pieces) == 0:
            hand_pieces = "-"
        sfen += hand_pieces

        # 手数
        sfen += " " + str(self.game_ply)
        return sfen

    def set_usi_position(self, position_command: str):
        """
        USIの"position"コマンドに従って局面をセットする。
        :param position_command: "position startpos moves 2h6h"
        :return:
        """
        items = position_command.rstrip().split()
        assert items.pop(0) == "position"
        if items[0] == "startpos":
            items.pop(0)
            self.set_hirate()
        elif items[0] == "sfen":
            items.pop(0)
            # position sfen lnsg... b - 3 moves 7g7f ...
            self.set_sfen(" ".join(items[:4]))
            del items[:4]
        else:
            raise NotImplementedError
        if len(items) > 0:  # 将棋所で初形だと"position startpos"で終わり
            assert items.pop(0) == "moves"
            for move_str in items:
                move = Move.from_usi_string(move_str)
                self.do_move(move)

    _ROTATE_PIECE_TABLE = np.array([Piece.NO_PIECE, Piece.W_PAWN, Piece.W_LANCE, Piece.W_KNIGHT,
                                    Piece.W_SILVER, Piece.W_BISHOP, Piece.W_ROOK, Piece.W_GOLD,
                                    Piece.W_KING, Piece.W_PRO_PAWN, Piece.W_PRO_LANCE, Piece.W_PRO_KNIGHT,
                                    Piece.W_PRO_SILVER, Piece.W_HORSE, Piece.W_DRAGON, Piece.W_QUEEN,
                                    Piece.NO_PIECE, Piece.B_PAWN, Piece.B_LANCE, Piece.B_KNIGHT,
                                    Piece.B_SILVER, Piece.B_BISHOP, Piece.B_ROOK, Piece.B_GOLD,
                                    Piece.B_KING, Piece.B_PRO_PAWN, Piece.B_PRO_LANCE, Piece.B_PRO_KNIGHT,
                                    Piece.B_PRO_SILVER, Piece.B_HORSE, Piece.B_DRAGON, Piece.B_QUEEN
                                    ], dtype=np.uint8)

    def _rotate_position(self) -> "Position":
        """
        逆の手番から見た盤面を生成する。
        盤面・持ち駒・手番を反転したインスタンスを生成。
        :return:
        """
        rot = Position()
        #rot.board[:] = Position._ROTATE_PIECE_TABLE[self.board[::-1]]
        rot.set_board(Position._ROTATE_PIECE_TABLE[self.board[::-1]])
        #rot.hand[:] = self.hand[::-1, :]
        rot.set_hand(self.hand[::-1, :].copy())
        rot.side_to_move = Color.invert(self.side_to_move)
        return rot
