"""
駒得評価のみで探索するプレイヤーの実装
"""
import random
import time
from typing import Dict, Optional, Tuple, List
import numpy as np

from logging import getLogger

logger = getLogger(__name__)

from .move import Piece, Color, Move
from .position import Position, PositionHelper
from .engine import Engine
from .usi_info_writer import UsiInfoWriter

# From Apery (WCSC26)
PIECE_VALUES = {
    Piece.NO_PIECE: 0,
    Piece.PAWN: 90,
    Piece.LANCE: 315,
    Piece.KNIGHT: 405,
    Piece.SILVER: 495,
    Piece.GOLD: 540,
    Piece.BISHOP: 855,
    Piece.ROOK: 990,
    Piece.PRO_PAWN: 540,
    Piece.PRO_LANCE: 540,
    Piece.PRO_KNIGHT: 540,
    Piece.PRO_SILVER: 540,
    Piece.HORSE: 945,
    Piece.DRAGON: 1395,
    Piece.KING: 15000,
    Piece.QUEEN: 0,
}

PIECE_VALUES_BOARD: np.ndarray = None
PIECE_VALUES_HAND: np.ndarray = None


def _make_value_table():
    global PIECE_VALUES_BOARD
    global PIECE_VALUES_HAND
    board_values = []
    # black
    for i in range(16):
        board_values.append(PIECE_VALUES[i])
    # white
    for i in range(16):
        board_values.append(-PIECE_VALUES[i])
    PIECE_VALUES_BOARD = np.array(board_values, dtype=np.int32)
    hand_values = []
    for i in range(1, 8):
        hand_values.append(PIECE_VALUES[i])
    PIECE_VALUES_HAND = np.array(hand_values, dtype=np.int32)


_make_value_table()


class MaterialPlayer(Engine):
    pos: Position
    depth: int
    best_move: Move
    best_move_table: Dict[int, Move]  # 16bit int
    nodes: int

    def __init__(self):
        self.pos = Position()
        self.depth = 1

    @property
    def name(self):
        return "NeneShogi Material"

    @property
    def author(self):
        return "select766"

    def get_options(self):
        return {"depth": "spin default 3 min 1 max 10"}

    def isready(self, options: Dict[str, str]):
        self.depth = int(options["depth"])
        self.best_move_table = {}

    def position(self, command: str):
        PositionHelper.set_usi_position(self.pos, command)

    def go(self, usi_info_writer: UsiInfoWriter, go_receive_time: float, btime: Optional[int] = None,
           wtime: Optional[int] = None, byoyomi: Optional[int] = None, binc: Optional[int] = None,
           winc: Optional[int] = None):
        self.best_move = None
        self.nodes = 0
        go_begin_time = time.time()
        for depth in range(1, self.depth + 1):
            val = self._search(depth, True, -60000, 60000)
            pv = self._retrieve_pv()
            go_elapsed_time = time.time() - go_begin_time
            usi_info_writer.write_pv(pv=pv, depth=depth, score_cp=int(val), time=int(go_elapsed_time * 1000),
                                     nodes=self.nodes)
            if self.best_move is None:
                return "resign"
        return self.best_move.to_usi_string()

    def _search(self, depth: int, root: bool, alpha: int, beta: int) -> int:
        pos = self.pos
        self.nodes += 1
        val = 0
        if depth > 0:
            move_list = self.pos.generate_move_list()
            if len(move_list) == 0:
                # mate
                val = -30000
            else:
                best_move = None
                random.shuffle(move_list)
                for move in move_list:
                    pos.do_move(move)
                    child_val = -self._search(depth - 1, False, -beta, -alpha)
                    pos.undo_move()
                    if child_val > alpha:
                        alpha = child_val
                        best_move = move
                    if alpha >= beta:
                        break
                if best_move is not None:
                    self.best_move_table[pos.key() % 65521] = best_move
                if root:
                    self.best_move = best_move
                return alpha
        else:
            # evaluate
            val = np.sum(PIECE_VALUES_BOARD[pos.get_board()])
            hand = pos.get_hand()
            val += np.sum(PIECE_VALUES_HAND * hand[0])
            val -= np.sum(PIECE_VALUES_HAND * hand[1])
            if pos.side_to_move() == Color.WHITE:
                val = -val
        return val

    def _retrieve_pv(self) -> List[Move]:
        pos = self.pos
        pv = []
        undo_count = 0
        for i in range(self.depth):  # 千日手で無限ループを回避するため
            move = self.best_move_table.get(pos.key() % 65521)
            if move is None:
                break
            pos.do_move(move)
            undo_count += 1
            pv.append(move)
        while undo_count > 0:
            pos.undo_move()
            undo_count -= 1
        return pv
