"""
DNN評価値関数による、1手読みプレイヤーの実装
"""
import random
from typing import Dict, Optional, List

from logging import getLogger

logger = getLogger(__name__)

import numpy as np
import chainer

from .position import Position, Color, Square, Piece, Move, PositionHelper
from yaneuraou import DNNConverter
from .engine import Engine
from .usi_info_writer import UsiInfoWriter
from .train_config import load_model
from . import util


class OneSearchPlayer(Engine):
    pos: Position
    model: chainer.Chain
    gpu: int
    dnn_converter: DNNConverter

    def __init__(self):
        self.pos = Position()
        self.model = None
        self.gpu = -1
        self.dnn_converter = DNNConverter(1, 1)

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
        PositionHelper.set_usi_position(self.pos, command)

    def _make_strategy(self, usi_info_writer: UsiInfoWriter):
        """
        1手展開した結果に対し、評価関数を呼び出して手を決定する
        :return:
        """
        if self.pos.is_mated():
            return Move.MOVE_RESIGN
        dnn_inputs = []
        move_list = self.pos.generate_move_list()
        for move in move_list:
            self.pos.do_move(move)
            dnn_inputs.append(self.dnn_converter.get_board_array(self.pos))
            self.pos.undo_move()
        with chainer.using_config("train", False):
            dnn_input = np.stack(dnn_inputs, axis=0)
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
    def go(self, usi_info_writer: UsiInfoWriter, go_receive_time: float, btime: Optional[int] = None,
           wtime: Optional[int] = None, byoyomi: Optional[int] = None, binc: Optional[int] = None,
           winc: Optional[int] = None):
        move = self._make_strategy(usi_info_writer)
        return move.to_usi_string()
