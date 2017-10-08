from typing import List, Tuple
import numpy as np
from .move import Move
cimport cython
cimport numpy as np

"""
手番定数
"""
cdef enum Color:
    BLACK = 0  # 先手
    WHITE = 1  # 後手
    COLOR_NB = 2


"""
駒定数
"""
cdef enum Piece:

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

cdef int piece_is_exist(int piece):
    """
    駒が存在するかどうか(空のマスでないか)を判定する
    :param piece:
    :return:
    """
    return piece > Piece.PIECE_ZERO

cdef int piece_is_color(int piece, int color):
    """
    駒が特定の色かどうか判定する
    :param piece:
    :param color:
    :return:
    """
    if piece == Piece.PIECE_ZERO:
        return False
    return piece // Piece.PIECE_WHITE == color

"""
マス定数
筋*9+段
1筋を0、1段を0に割り当てる
"""
cdef enum Square:
    SQ_NB = 81


cdef int square_from_file_rank(int file, int rank):
    return file * 9 + rank

cdef int square_from_file_rank_if_valid(int file, int rank):
    cdef int valid = file >= 0 and file < 9 and rank >= 0 and rank < 9
    if not valid:
        return -1
    sq = file * 9 + rank
    return sq


cdef int square_file_of(int sq):
    """
    筋を返す
    :param sq:
    :return:
    """
    return sq // 9

cdef int square_rank_of(int sq):
    """
    段を返す
    :param sq:
    :return:
    """
    return sq % 9

cdef int _CHECK_ATTACK_DIRS[8][2]
_CHECK_ATTACK_DIRS[0][:] = (-1, -1)
_CHECK_ATTACK_DIRS[1][:] = (0, -1)
_CHECK_ATTACK_DIRS[2][:] = (1, -1)
_CHECK_ATTACK_DIRS[3][:] = (-1, 0)
_CHECK_ATTACK_DIRS[4][:] = (1, 0)
_CHECK_ATTACK_DIRS[5][:] = (-1, 1)
_CHECK_ATTACK_DIRS[6][:] = (0, 1)
_CHECK_ATTACK_DIRS[7][:] = (1, 1)

# 先手玉の左上、上、右上、…に存在すると、王手を構成する後手の駒(短い利き)。
cdef int _CHECK_SHORT_ATTACK_PIECES[8][13]
_CHECK_SHORT_ATTACK_PIECES[0][:11]=    [Piece.W_SILVER, Piece.W_BISHOP, Piece.W_GOLD, Piece.W_KING, Piece.W_PRO_PAWN, Piece.W_PRO_LANCE,
     Piece.W_PRO_KNIGHT, Piece.W_PRO_SILVER, Piece.W_HORSE, Piece.W_DRAGON, -1]  # 左上
_CHECK_SHORT_ATTACK_PIECES[1][:13]=    [Piece.W_PAWN, Piece.W_LANCE, Piece.W_SILVER, Piece.W_ROOK, Piece.W_GOLD, Piece.W_KING, Piece.W_PRO_PAWN,
     Piece.W_PRO_LANCE,
     Piece.W_PRO_KNIGHT, Piece.W_PRO_SILVER, Piece.W_HORSE, Piece.W_DRAGON,-1]  # 上
_CHECK_SHORT_ATTACK_PIECES[2][:11]=    [Piece.W_SILVER, Piece.W_BISHOP, Piece.W_GOLD, Piece.W_KING, Piece.W_PRO_PAWN, Piece.W_PRO_LANCE,
     Piece.W_PRO_KNIGHT, Piece.W_PRO_SILVER, Piece.W_HORSE, Piece.W_DRAGON,-1]  # 右上
_CHECK_SHORT_ATTACK_PIECES[3][:10]=    [Piece.W_ROOK, Piece.W_GOLD, Piece.W_KING, Piece.W_PRO_PAWN, Piece.W_PRO_LANCE,
     Piece.W_PRO_KNIGHT, Piece.W_PRO_SILVER, Piece.W_HORSE, Piece.W_DRAGON,-1]  # 左
_CHECK_SHORT_ATTACK_PIECES[4][:10]=    [Piece.W_ROOK, Piece.W_GOLD, Piece.W_KING, Piece.W_PRO_PAWN, Piece.W_PRO_LANCE,
     Piece.W_PRO_KNIGHT, Piece.W_PRO_SILVER, Piece.W_HORSE, Piece.W_DRAGON,-1]  # 右
_CHECK_SHORT_ATTACK_PIECES[5][:6]=    [Piece.W_SILVER, Piece.W_BISHOP, Piece.W_KING, Piece.W_HORSE, Piece.W_DRAGON,-1]  # 左下
_CHECK_SHORT_ATTACK_PIECES[6][:10]=    [Piece.W_ROOK, Piece.W_GOLD, Piece.W_KING, Piece.W_PRO_PAWN, Piece.W_PRO_LANCE,
     Piece.W_PRO_KNIGHT, Piece.W_PRO_SILVER, Piece.W_HORSE, Piece.W_DRAGON,-1]  # 下
_CHECK_SHORT_ATTACK_PIECES[7][:6]=    [Piece.W_SILVER, Piece.W_BISHOP, Piece.W_KING, Piece.W_HORSE, Piece.W_DRAGON,-1]  # 右下

# 先手玉の左上、上、右上、…に存在すると、王手を構成する後手の駒(長い利き)。
cdef int _CHECK_LONG_ATTACK_PIECES[8][4]

_CHECK_LONG_ATTACK_PIECES[0][:3]=    [Piece.W_BISHOP, Piece.W_HORSE,-1]  # 左上
_CHECK_LONG_ATTACK_PIECES[1][:4]=    [Piece.W_LANCE, Piece.W_ROOK, Piece.W_DRAGON,-1]  # 上
_CHECK_LONG_ATTACK_PIECES[2][:3]=    [Piece.W_BISHOP, Piece.W_HORSE,-1]  # 右上
_CHECK_LONG_ATTACK_PIECES[3][:3]=    [Piece.W_ROOK, Piece.W_DRAGON,-1]  # 左
_CHECK_LONG_ATTACK_PIECES[4][:3]=    [Piece.W_ROOK, Piece.W_DRAGON,-1]  # 右
_CHECK_LONG_ATTACK_PIECES[5][:3]=    [Piece.W_BISHOP, Piece.W_HORSE,-1]  # 左下
_CHECK_LONG_ATTACK_PIECES[6][:3]=    [Piece.W_ROOK, Piece.W_DRAGON,-1]  # 下
_CHECK_LONG_ATTACK_PIECES[7][:3]=    [Piece.W_BISHOP, Piece.W_HORSE,-1]  # 右下


ctypedef np.uint8_t DTYPE_t
@cython.boundscheck(False)
def _in_check_black(np.ndarray[DTYPE_t, ndim=1] board) -> bool:
    """
    先手が王手された状態かどうかをチェックする。
    先手が指して、後手番状態で呼び出すことも可能。この場合、王手放置のチェックとなる。
    :return:
    """

    # 先手玉からみて各方向に後手の駒があれば、王手されていることになる。
    # 例えば、先手玉の1つ下に後手歩があれば王手。
    # 先手玉の右下に、他の駒に遮られずに角があれば王手。
    # 長い利きの場合、途中のマスがすべて空でなければならない。
    cdef int black_king_sq = 0
    cdef int sq
    for sq in range(Square.SQ_NB):
        if board[sq] == Piece.B_KING:
            black_king_sq = sq
            break
    cdef int black_king_file = square_file_of(black_king_sq)
    cdef int black_king_rank = square_rank_of(black_king_sq)
    cdef int x, y, attacker_file, attacker_rank, attacker_sq
    cdef int attacker_piece
    cdef int pt_i
    cdef int pt_tmp
    for dir_i in range(8):
        x, y = _CHECK_ATTACK_DIRS[dir_i]
        attacker_file = black_king_file + x
        attacker_rank = black_king_rank + y
        attacker_sq = square_from_file_rank_if_valid(attacker_file, attacker_rank)
        if attacker_sq < 0:
            continue

        attacker_piece = board[attacker_sq]
        for pt_i in range(13):
            pt_tmp = _CHECK_SHORT_ATTACK_PIECES[dir_i][pt_i]
            if pt_tmp < 0:
                break
            if attacker_piece == pt_tmp:
                # 短い利きが有効
                return True
        # マスが空なら、長い利きをチェック
        while True:
            if piece_is_exist(attacker_piece):
                # 空白以外の駒があるなら利きが切れる
                break

            attacker_file += x
            attacker_rank += y
            attacker_sq = square_from_file_rank_if_valid(attacker_file, attacker_rank)
            if attacker_sq < 0:
                break
            attacker_piece = board[attacker_sq]
            for pt_i in range(4):
                pt_tmp = _CHECK_LONG_ATTACK_PIECES[dir_i][pt_i]
                if pt_tmp < 0:
                    break
                if attacker_piece == pt_tmp:
                    # 長い利きが有効
                    return True

    # 桂馬の利きチェック
    for x in range(-1, 2, 2):#[-1, 1]:
        attacker_file = black_king_file + x
        attacker_rank = black_king_rank - 2
        attacker_sq = square_from_file_rank_if_valid(attacker_file, attacker_rank)
        if attacker_sq < 0:
            continue

        attacker_piece = board[attacker_sq]
        if attacker_piece == Piece.W_KNIGHT:
            # 桂馬がいる
            return True

    return False

cdef struct Vec:
    int x
    int y

_SHORT_ATTACK_TABLE_py = [[],
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
cdef int _SHORT_ATTACK_TABLE[15][8][2]
cdef int _SHORT_ATTACK_TABLE_LEN[15]
for i, d in enumerate(_SHORT_ATTACK_TABLE_py):
    _SHORT_ATTACK_TABLE_LEN[i] = len(d)
    for j, (x, y) in enumerate(d):
        _SHORT_ATTACK_TABLE[i][j] = (x, y)

cdef int _MAX_NON_PROMOTE_RANK_TABLE[15]
_MAX_NON_PROMOTE_RANK_TABLE[:] = [0,
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

_LONG_ATTACK_TABLE_py = [[],
                      [],  # 歩
                      [(0, -1)],  # 香
                      [],  # 桂
                      [],  # 銀
                      [(-1, -1), (1, -1), (-1, 1), (1, 1)],  # 角
                      [(0, -1), (-1, 0), (1, 0), (0, 1)],  # 飛
                      [],  # 金
                      [],  # 玉
                      [],  # と
                      [],  # 成香
                      [],  # 成桂
                      [],  # 成銀
                      [(-1, -1), (1, -1), (-1, 1), (1, 1)],  # 馬
                      [(0, -1), (-1, 0), (1, 0), (0, 1)],  # 竜
                      ]

cdef int _LONG_ATTACK_TABLE[15][4][2]
cdef int _LONG_ATTACK_TABLE_LEN[15]
for i, d in enumerate(_LONG_ATTACK_TABLE_py):
    _LONG_ATTACK_TABLE_LEN[i] = len(d)
    for j, (x, y) in enumerate(d):
        _LONG_ATTACK_TABLE[i][j] = (x, y)

_MAX_DROP_RANK_TABLE = [0, 1, 1, 2, 0, 0, 0, 0]

@cython.boundscheck(False)
def _generate_move_move(np.ndarray[DTYPE_t, ndim=1] board) -> List[Move]:
    """
    盤上の駒を動かす手をすべて生成する。
    先手番を前提とする。
    ただし、香車の2段目・歩・角・飛の不成りおよび行き場のない駒を生じる手は除く。
    :return:
    """
    possible_moves = []
    cdef int from_file, from_rank, from_sq, from_piece, can_promote, to_file, to_rank, to_sq, to_piece
    cdef int x, y
    for from_file in range(9):
        for from_rank in range(9):
            from_sq = square_from_file_rank(from_file, from_rank)
            from_piece = board[from_sq]
            if not piece_is_color(from_piece, Color.BLACK):
                continue
            can_promote = from_piece <= Piece.B_ROOK
            short_attacks = _SHORT_ATTACK_TABLE[from_piece]  # type: List[Tuple[int, int]]
            max_non_promote_rank = _MAX_NON_PROMOTE_RANK_TABLE[from_piece]  # type: int
            # 短い利きの処理
            for short_attack_i in range(_SHORT_ATTACK_TABLE_LEN[from_piece]):
                x = _SHORT_ATTACK_TABLE[from_piece][short_attack_i][0]
                y = _SHORT_ATTACK_TABLE[from_piece][short_attack_i][1]
                to_file = from_file + x
                to_rank = from_rank + y
                # 盤内確認
                to_sq = square_from_file_rank_if_valid(to_file, to_rank)
                if to_sq < 0:
                    continue
                to_piece = board[to_sq]
                # 自分の駒があるところには進めない
                if not piece_is_color(to_piece, Color.BLACK):
                    if to_rank >= max_non_promote_rank:
                        # 行き場のない駒にはならない(&無意味な不成ではない)
                        possible_moves.append(Move.make_move(from_sq, to_sq, False))
                    if can_promote and (from_rank < 3 or to_rank < 3):
                        # 成れる駒で、成る条件を満たす
                        possible_moves.append(Move.make_move(from_sq, to_sq, True))
            # 長い利きの処理
            for long_attack_i in range(_LONG_ATTACK_TABLE_LEN[from_piece]):
                x = _LONG_ATTACK_TABLE[from_piece][long_attack_i][0]
                y = _LONG_ATTACK_TABLE[from_piece][long_attack_i][1]
                to_file = from_file
                to_rank = from_rank
                while True:
                    to_file += x
                    to_rank += y
                    to_sq = square_from_file_rank_if_valid(to_file, to_rank)
                    if to_sq < 0:
                        break
                    to_piece = board[to_sq]
                    # 自分の駒があるところには進めない
                    if piece_is_color(to_piece, Color.BLACK):
                        break
                    if to_rank >= max_non_promote_rank and from_rank >= max_non_promote_rank:
                        # 成って損がないのに成らない状況以外(角・飛)
                        possible_moves.append(Move.make_move(from_sq, to_sq, False))
                    if can_promote and (from_rank < 3 or to_rank < 3):
                        # 成れる駒で、成る条件を満たす
                        possible_moves.append(Move.make_move(from_sq, to_sq, True))
                    if piece_is_exist(to_piece):
                        # 白駒があるので、これ以上進めない
                        break

    return possible_moves
