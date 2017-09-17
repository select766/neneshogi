"""
DNN評価値関数による、1手読みプレイヤーの実装
"""
import random
from typing import Dict, Optional, List

from logging import getLogger

logger = getLogger(__name__)

import numpy as np
import chainer

from .position import Position, Color, Square, Piece, Move
from .engine import Engine
from .usi_info_writer import UsiInfoWriter
from .train_config import load_model
from . import util


class OneSearchPlayer(Engine):
    pos: Position
    model: chainer.Chain
    gpu: int

    def __init__(self):
        self.pos = Position()
        self.model = None
        self.gpu = -1

    @property
    def name(self):
        return "NeneShogi OneSearch"

    @property
    def author(self):
        return "select766"

    def get_options(self):
        return {"model_path": "filename default <empty>",
                "gpu": "spin default -1 min -1 max 0"}

    def isready(self, options: Dict[str, str]):
        self.gpu = int(options["gpu"])
        self.model = load_model(options["model_path"])
        if self.gpu >= 0:
            chainer.cuda.get_device_from_id(self.gpu).use()
            self.model.to_gpu()

    def position(self, command: str):
        self.pos.set_usi_position(command)

    def _make_dnn_input(self, pos: Position):
        """
        与えられたPositionからDNNへの入力行列を生成する。
        :return:
        """
        # 常に現在の手番が先手となるように盤面を与える
        pos_from_side = pos
        if pos_from_side.side_to_move == Color.WHITE:
            pos_from_side = pos_from_side._rotate_position()
        ary = np.zeros((61, 81), dtype=np.float32)
        # 盤上の駒
        for sq in range(Square.SQ_NB):
            piece = pos_from_side.board[sq]
            ch = -1
            if piece >= Piece.W_PAWN:
                ch = piece - Piece.W_PAWN + 14
            elif piece >= Piece.B_PAWN:
                ch = piece - Piece.B_PAWN
            if ch >= 0:
                ary[ch, sq] = 1.0
        # 持ち駒
        for color in range(Color.COLOR_NB):
            for i in range(Piece.PIECE_HAND_NB - Piece.PIECE_HAND_ZERO):
                hand_count = pos_from_side.hand[color][i]
                ch = color * 7 + 28 + i
                ary[ch, :] = hand_count
        # 段・筋
        for sq in range(Square.SQ_NB):
            ary[Square.rank_of(sq) + 42, sq] = 1.0
            ary[Square.file_of(sq) + 51, sq] = 1.0
        # 定数1
        ary[60, :] = 1.0
        return ary.reshape((1, 61, 9, 9))

    def _make_strategy(self, usi_info_writer: UsiInfoWriter, move_list: List[Move]):
        """
        1手展開した結果に対し、評価関数を呼び出して手を決定する
        :return:
        """
        dnn_inputs = []
        for move in move_list:
            undo_info = self.pos.do_move(move)
            dnn_inputs.append(self._make_dnn_input(self.pos))
            self.pos.undo_move(undo_info)
        with chainer.using_config("train", False):
            dnn_input = np.concatenate(dnn_inputs, axis=0)
            if self.gpu >= 0:
                dnn_input = chainer.cuda.to_gpu(dnn_input)
            model_output_var_move, model_output_var_value = self.model.forward(dnn_input)
            model_output = chainer.cuda.to_cpu(model_output_var_value.data)  # type: np.ndarray
        # 1手先の局面なので、相手から見た評価値が返っている
        my_score = -model_output
        max_move_index = int(np.argmax(my_score))
        max_move = move_list[max_move_index]
        max_move_score = my_score[max_move_index]
        usi_info_writer.write_pv(pv=[max_move], depth=1, score_cp=int(max_move_score * 600))
        return max_move

    @util.release_gpu_memory_pool
    def go(self, usi_info_writer: UsiInfoWriter, btime: Optional[int] = None, wtime: Optional[int] = None,
           byoyomi: Optional[int] = None, binc: Optional[int] = None, winc: Optional[int] = None):
        move_list = self.pos.generate_move_list()
        if len(move_list) == 0:
            return "resign"

        move = self._make_strategy(usi_info_writer, move_list)
        return move.to_usi_string()
