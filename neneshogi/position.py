"""
盤面を表現するクラス
関数名等は、可能な限りやねうら王に準拠
"""
from typing import List, Tuple

import numpy as np
from .move import Color, Piece, Square, Move, UndoMoveInfo
from . import position_acc


class Position:
    """
    盤面を表現するクラス
    """
    board: np.ndarray  # 盤面(81要素,np.uint8)
    hand: np.ndarray  # 持ち駒((2,7)要素,np.uint8)
    side_to_move: int  # 手番
    game_ply: int  # 手数(開始局面で1)
    # 平手開始局面の盤面
    HIRATE_BOARD = np.array([18, 0, 17, 0, 0, 0, 1, 0, 2,
                             19, 21, 17, 0, 0, 0, 1, 6, 3,
                             20, 0, 17, 0, 0, 0, 1, 0, 4,
                             23, 0, 17, 0, 0, 0, 1, 0, 7,
                             24, 0, 17, 0, 0, 0, 1, 0, 8,
                             23, 0, 17, 0, 0, 0, 1, 0, 7,
                             20, 0, 17, 0, 0, 0, 1, 0, 4,
                             19, 22, 17, 0, 0, 0, 1, 5, 3,
                             18, 0, 17, 0, 0, 0, 1, 0, 2, ], dtype=np.uint8)

    def __init__(self):
        self.board = np.zeros((Square.SQ_NB,), dtype=np.uint8)
        self.hand = np.zeros((Color.COLOR_NB, (Piece.PIECE_HAND_NB - Piece.PIECE_HAND_ZERO)), dtype=np.uint8)
        self.side_to_move = Color.BLACK
        self.game_ply = 1

    def set_hirate(self):
        """
        平手開始局面をセットする
        :return:
        """
        self.board[:] = Position.HIRATE_BOARD
        self.hand[:] = 0
        self.side_to_move = Color.BLACK
        self.game_ply = 1

    def set_sfen(self, sfen: str) -> None:
        """
        SFEN形式の局面をセットする。
        :param sfen:
        :return:
        """
        board_str, color_str, hand_str, ply_str = sfen.split()
        board_ranks = board_str.split("/")
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
                        self.board[sq] = Piece.NO_PIECE
                        file_from_left += 1
                else:
                    # 駒
                    piece = Piece.piece_from_char(token)
                    if is_promote:
                        piece += Piece.PIECE_PROMOTE
                    sq = Square.from_file_rank(8 - file_from_left, rank)
                    self.board[sq] = piece
                    file_from_left += 1
                is_promote = False
            assert file_from_left == 9

        self.hand[:] = 0
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
                    self.hand[piece_color, Piece.raw_pt_from_piece(piece) - Piece.PIECE_HAND_ZERO] = num_piece
                    num_piece_str = ""

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
        for y in range(9):
            if y > 0:
                sfen += "/"
            blank_len = 0
            for x in range(9):
                sq = (8 - x) * 9 + y
                piece = self.board[sq]
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
        for color in range(Color.COLOR_NB):
            hand_for_color = self.hand[color]
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

    def do_move(self, move: Move) -> UndoMoveInfo:
        """
        手を指して局面を進める
        :param move:
        :return: 局面を戻すために必要な情報
        """
        undo_move_info = UndoMoveInfo(self)
        if move.is_drop:
            # 駒打ち
            # 持ち駒を減らす
            self.hand[self.side_to_move, move.move_dropped_piece - Piece.PIECE_HAND_ZERO] -= 1
            # 駒を置く
            piece = move.move_dropped_piece
            if self.side_to_move == Color.WHITE:
                piece += Piece.PIECE_WHITE  # move_dropped_pieceは駒種
            self.board[move.move_to] = piece
        else:
            # 駒の移動
            piece = self.board[move.move_from]
            captured_piece = self.board[move.move_to]
            if captured_piece != Piece.PIECE_ZERO:
                # 持ち駒を増やす
                self.hand[self.side_to_move, Piece.raw_pt_from_piece(captured_piece) - Piece.PIECE_HAND_ZERO] += 1
            self.board[move.move_from] = Piece.PIECE_ZERO
            if move.is_promote:
                piece += Piece.PIECE_PROMOTE
            self.board[move.move_to] = piece

        # 手番を逆転
        self.side_to_move = Color.invert(self.side_to_move)

        # 手数を加算
        self.game_ply += 1
        return undo_move_info

    def undo_move(self, undo_move_info: UndoMoveInfo) -> None:
        """
        局面を戻す
        :param undo_move_info:
        :return:
        """
        self.game_ply -= 1
        self.side_to_move = Color.invert(self.side_to_move)
        self.board[:] = undo_move_info.board
        self.hand[:] = undo_move_info.hand

    def eq_board(self, other: "Position") -> bool:
        """
        駒の配置・持ち駒・手番が一致するかどうか調べる。
        手数・指し手の履歴は考慮しない。
        :param other:
        :return:
        """
        if self.side_to_move != other.side_to_move:
            return False
        if not np.all(self.hand == other.hand):
            return False
        if not np.all(self.board == other.board):
            return False
        return True

    def generate_move_list(self) -> List[Move]:
        """
        合法手のリストを生成する
        :return:
        """

        if self.side_to_move == Color.BLACK:
            return self._generate_move_list_black()
        else:
            # 後手番の場合は盤面を回転
            rot_pos = self._rotate_position()
            rot_move_list = rot_pos._generate_move_list_black()
            move_list = []
            for rot_move in rot_move_list:
                to_sq = Square.SQ_NB - 1 - rot_move.move_to
                if rot_move.is_drop:
                    move = Move.make_move_drop(rot_move.move_dropped_piece, to_sq)
                else:
                    from_sq = Square.SQ_NB - 1 - rot_move.move_from
                    move = Move.make_move(from_sq, to_sq, rot_move.is_promote)
                move_list.append(move)
            return move_list

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
        rot.board[:] = Position._ROTATE_PIECE_TABLE[self.board[::-1]]
        rot.hand[:] = self.hand[::-1, :]
        rot.side_to_move = Color.invert(self.side_to_move)
        return rot

    def _generate_move_list_black(self) -> List[Move]:
        """
        先手番の場合に、合法手のリストを生成する
        :return:
        """
        # 盤上の駒を利きの範囲で動かすすべての手、空きマスに持ち駒を打つすべての手を生成したのち、禁じ手を削除する。
        moves = self._generate_move_move() + self._generate_move_drop()

        legal_moves = []
        for move in moves:
            legal = True
            # 王手放置チェック
            undo_info = self.do_move(move)
            if self._in_check_black():
                # 後手番になっているのに先手が王手をかけられている
                legal = False
            # 打ち歩詰めチェック
            if legal and move.is_drop and move.move_dropped_piece == Piece.PAWN:
                # 王手放置のときにチェックすると、玉を取る手が生成されてバグる
                # 現在の手番(後手)が詰んでいるとき、打ち歩詰め
                # 玉の頭に打った時だけ判定すればよい
                white_king_check_pos = move.move_to - 1  # 1段目に打つ手は生成しないので、必ず盤内
                if self.board[white_king_check_pos] == Piece.W_KING:
                    if len(self.generate_move_list()) == 0:
                        legal = False
            self.undo_move(undo_info)
            if legal:
                legal_moves.append(move)

        return legal_moves

    _SHORT_ATTACK_TABLE = [None,
                           [(0, -1)],  # 歩
                           [],  # 香
                           [(-1, -2), (1, -2)],  # 桂
                           [(-1, -1), (0, -1), (1, -1), (-1, 1), (1, 1)],  # 銀
                           [],  # 角
                           [],  # 飛
                           [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (0, 1)],  # 金
                           [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)],  # 玉
                           [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (0, 1)],  # と
                           [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (0, 1)],  # 成香
                           [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (0, 1)],  # 成桂
                           [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (0, 1)],  # 成銀
                           [(0, -1), (-1, 0), (1, 0), (0, 1)],  # 馬
                           [(-1, -1), (1, -1), (-1, 1), (1, 1)],  # 竜
                           ]
    _MAX_NON_PROMOTE_RANK_TABLE = [0,
                                   3,  # 歩(必ず成る)
                                   2,  # 香(2段目では必ず成る)
                                   2,  # 桂
                                   0,  # 銀
                                   3,  # 角(必ず成る)
                                   3,  # 飛(必ず成る)
                                   0,  # 金
                                   0,
                                   0,
                                   0,
                                   0,
                                   0,
                                   0,
                                   0,
                                   ]
    _LONG_ATTACK_TABLE = [None,
                          None,  # 歩
                          [(0, -1)],  # 香
                          None,  # 桂
                          None,  # 銀
                          [(-1, -1), (1, -1), (-1, 1), (1, 1)],  # 角
                          [(0, -1), (-1, 0), (1, 0), (0, 1)],  # 飛
                          None,  # 金
                          None,  # 玉
                          None,  # と
                          None,  # 成香
                          None,  # 成桂
                          None,  # 成銀
                          [(-1, -1), (1, -1), (-1, 1), (1, 1)],  # 馬
                          [(0, -1), (-1, 0), (1, 0), (0, 1)],  # 竜
                          ]
    _MAX_DROP_RANK_TABLE = [0, 1, 1, 2, 0, 0, 0, 0]

    def _generate_move_move(self) -> List[Move]:
        return position_acc._generate_move_move(self.board)

    def _generate_move_move_native(self) -> List[Move]:
        """
        盤上の駒を動かす手をすべて生成する。
        先手番を前提とする。
        ただし、香車の2段目・歩・角・飛の不成りおよび行き場のない駒を生じる手は除く。
        :return:
        """
        possible_moves = []
        board = self.board
        for from_file in range(9):
            for from_rank in range(9):
                from_sq = Square.from_file_rank(from_file, from_rank)
                from_piece = board[from_sq]
                if not Piece.is_color(from_piece, Color.BLACK):
                    continue
                can_promote = from_piece <= Piece.B_ROOK
                short_attacks = Position._SHORT_ATTACK_TABLE[from_piece]  # type: List[Tuple[int, int]]
                max_non_promote_rank = Position._MAX_NON_PROMOTE_RANK_TABLE[from_piece]  # type: int
                long_attacks = Position._LONG_ATTACK_TABLE[from_piece]  # type: List[Tuple[int, int]]
                # 短い利きの処理
                for x, y in short_attacks:
                    to_file = from_file + x
                    to_rank = from_rank + y
                    # 盤内確認
                    to_sq, valid = Square.from_file_rank_if_valid(to_file, to_rank)
                    if not valid:
                        continue
                    to_piece = board[to_sq]
                    # 自分の駒があるところには進めない
                    if not Piece.is_color(to_piece, Color.BLACK):
                        if to_rank >= max_non_promote_rank:
                            # 行き場のない駒にはならない(&無意味な不成ではない)
                            possible_moves.append(Move.make_move(from_sq, to_sq, False))
                        if can_promote and (from_rank < 3 or to_rank < 3):
                            # 成れる駒で、成る条件を満たす
                            possible_moves.append(Move.make_move(from_sq, to_sq, True))
                # 長い利きの処理
                if long_attacks is not None:
                    for x, y in long_attacks:
                        to_file = from_file
                        to_rank = from_rank
                        while True:
                            to_file += x
                            to_rank += y
                            to_sq, valid = Square.from_file_rank_if_valid(to_file, to_rank)
                            if not valid:
                                break
                            to_piece = board[to_sq]
                            # 自分の駒があるところには進めない
                            if Piece.is_color(to_piece, Color.BLACK):
                                break
                            if to_rank >= max_non_promote_rank and from_rank >= max_non_promote_rank:
                                # 成って損がないのに成らない状況以外(角・飛)
                                possible_moves.append(Move.make_move(from_sq, to_sq, False))
                            if can_promote and (from_rank < 3 or to_rank < 3):
                                # 成れる駒で、成る条件を満たす
                                possible_moves.append(Move.make_move(from_sq, to_sq, True))
                            if Piece.is_exist(to_piece):
                                # 白駒があるので、これ以上進めない
                                break

        return possible_moves

    def _generate_move_drop(self) -> List[Move]:
        return position_acc._generate_move_drop(self.board, self.hand)

    def _generate_move_drop_native(self) -> List[Move]:
        """
        駒を打つ手をすべて生成する。
        先手番を前提とする。
        ただし、二歩・行き場のない駒を生じる手は除く。
        :return:
        """
        possible_moves = []
        board = self.board
        hand = self.hand[Color.BLACK]
        hand_pt_max_drop_ranks = []  # type: List[Tuple[int, int]]
        for pt in range(Piece.PIECE_HAND_ZERO, Piece.PIECE_HAND_NB):
            if hand[pt - Piece.PIECE_HAND_ZERO] > 0:
                hand_pt_max_drop_ranks.append((pt, Position._MAX_DROP_RANK_TABLE[pt]))
        if len(hand_pt_max_drop_ranks) == 0:
            # 持ち駒がなければこれ以上の処理は不要
            return []

        # 二歩を避けるため、歩がすでにある筋を列挙
        pawn_files = []
        for to_file in range(9):
            for to_rank in range(9):
                to_sq = Square.from_file_rank(to_file, to_rank)
                to_piece = board[to_sq]
                if to_piece == Piece.B_PAWN:
                    pawn_files.append(to_file)
                    break

        for to_file in range(9):
            for to_rank in range(9):
                to_sq = Square.from_file_rank(to_file, to_rank)
                to_piece = board[to_sq]
                if Piece.is_exist(to_piece):
                    # 駒のある場所には打てない
                    continue
                for pt, max_drop_rank in hand_pt_max_drop_ranks:
                    if pt == Piece.B_PAWN and to_file in pawn_files:
                        # 二歩
                        continue
                    if to_rank < max_drop_rank:
                        continue
                    possible_moves.append(Move.make_move_drop(pt, to_sq))

        return possible_moves

    _CHECK_ATTACK_DIRS = [(-1, -1), (0, -1), (1, -1),
                          (-1, 0), (1, 0),
                          (-1, 1), (0, 1), (1, 1)]
    # 先手玉の左上、上、右上、…に存在すると、王手を構成する後手の駒(短い利き)。
    _CHECK_SHORT_ATTACK_PIECES = [
        [Piece.W_SILVER, Piece.W_BISHOP, Piece.W_GOLD, Piece.W_KING, Piece.W_PRO_PAWN, Piece.W_PRO_LANCE,
         Piece.W_PRO_KNIGHT, Piece.W_PRO_SILVER, Piece.W_HORSE, Piece.W_DRAGON],  # 左上
        [Piece.W_PAWN, Piece.W_LANCE, Piece.W_SILVER, Piece.W_ROOK, Piece.W_GOLD, Piece.W_KING, Piece.W_PRO_PAWN,
         Piece.W_PRO_LANCE,
         Piece.W_PRO_KNIGHT, Piece.W_PRO_SILVER, Piece.W_HORSE, Piece.W_DRAGON],  # 上
        [Piece.W_SILVER, Piece.W_BISHOP, Piece.W_GOLD, Piece.W_KING, Piece.W_PRO_PAWN, Piece.W_PRO_LANCE,
         Piece.W_PRO_KNIGHT, Piece.W_PRO_SILVER, Piece.W_HORSE, Piece.W_DRAGON],  # 右上
        [Piece.W_ROOK, Piece.W_GOLD, Piece.W_KING, Piece.W_PRO_PAWN, Piece.W_PRO_LANCE,
         Piece.W_PRO_KNIGHT, Piece.W_PRO_SILVER, Piece.W_HORSE, Piece.W_DRAGON],  # 左
        [Piece.W_ROOK, Piece.W_GOLD, Piece.W_KING, Piece.W_PRO_PAWN, Piece.W_PRO_LANCE,
         Piece.W_PRO_KNIGHT, Piece.W_PRO_SILVER, Piece.W_HORSE, Piece.W_DRAGON],  # 右
        [Piece.W_SILVER, Piece.W_BISHOP, Piece.W_KING, Piece.W_HORSE, Piece.W_DRAGON],  # 左下
        [Piece.W_ROOK, Piece.W_GOLD, Piece.W_KING, Piece.W_PRO_PAWN, Piece.W_PRO_LANCE,
         Piece.W_PRO_KNIGHT, Piece.W_PRO_SILVER, Piece.W_HORSE, Piece.W_DRAGON],  # 下
        [Piece.W_SILVER, Piece.W_BISHOP, Piece.W_KING, Piece.W_HORSE, Piece.W_DRAGON],  # 右下
    ]
    # 先手玉の左上、上、右上、…に存在すると、王手を構成する後手の駒(長い利き)。
    _CHECK_LONG_ATTACK_PIECES = [
        [Piece.W_BISHOP, Piece.W_HORSE],  # 左上
        [Piece.W_LANCE, Piece.W_ROOK, Piece.W_DRAGON],  # 上
        [Piece.W_BISHOP, Piece.W_HORSE],  # 右上
        [Piece.W_ROOK, Piece.W_DRAGON],  # 左
        [Piece.W_ROOK, Piece.W_DRAGON],  # 右
        [Piece.W_BISHOP, Piece.W_HORSE],  # 左下
        [Piece.W_ROOK, Piece.W_DRAGON],  # 下
        [Piece.W_BISHOP, Piece.W_HORSE],  # 右下
    ]

    def in_check(self) -> bool:
        """
        手番側が王手されている状態かどうかをチェックする。
        :return:
        """
        if self.side_to_move == Color.BLACK:
            return self._in_check_black()
        else:
            # 後手番の場合は盤面を回転
            rot_pos = self._rotate_position()
            return rot_pos._in_check_black()

    def _in_check_black(self) -> bool:
        return position_acc._in_check_black(self.board)

    def _in_check_black_native(self) -> bool:
        """
        先手が王手された状態かどうかをチェックする。
        先手が指して、後手番状態で呼び出すことも可能。この場合、王手放置のチェックとなる。
        :return:
        """

        # 先手玉からみて各方向に後手の駒があれば、王手されていることになる。
        # 例えば、先手玉の1つ下に後手歩があれば王手。
        # 先手玉の右下に、他の駒に遮られずに角があれば王手。
        # 長い利きの場合、途中のマスがすべて空でなければならない。
        black_king_sq = 0
        board = self.board
        for sq in range(Square.SQ_NB):
            if board[sq] == Piece.B_KING:
                black_king_sq = sq
                break
        black_king_file = Square.file_of(black_king_sq)
        black_king_rank = Square.rank_of(black_king_sq)
        for dir_i in range(8):
            x, y = Position._CHECK_ATTACK_DIRS[dir_i]
            attacker_file = black_king_file + x
            attacker_rank = black_king_rank + y
            attacker_sq, valid = Square.from_file_rank_if_valid(attacker_file, attacker_rank)
            if not valid:
                continue

            attacker_piece = board[attacker_sq]
            if attacker_piece in Position._CHECK_SHORT_ATTACK_PIECES[dir_i]:
                # 短い利きが有効
                return True
            # マスが空なら、長い利きをチェック
            long_attack_pieces = Position._CHECK_LONG_ATTACK_PIECES[dir_i]
            while True:
                if Piece.is_exist(attacker_piece):
                    # 空白以外の駒があるなら利きが切れる
                    break

                attacker_file += x
                attacker_rank += y
                attacker_sq, valid = Square.from_file_rank_if_valid(attacker_file, attacker_rank)
                if not valid:
                    break
                attacker_piece = board[attacker_sq]
                if attacker_piece in long_attack_pieces:
                    # 長い利きが有効
                    return True

        # 桂馬の利きチェック
        for x in [-1, 1]:
            attacker_file = black_king_file + x
            attacker_rank = black_king_rank - 2
            attacker_sq, valid = Square.from_file_rank_if_valid(attacker_file, attacker_rank)
            if not valid:
                continue

            attacker_piece = board[attacker_sq]
            if attacker_piece == Piece.W_KNIGHT:
                # 桂馬がいる
                return True

        return False
