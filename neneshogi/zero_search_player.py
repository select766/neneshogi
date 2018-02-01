"""
DNN方策関数による、0手読みプレイヤーの実装
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


class ZeroSearchPlayer(Engine):
    pos: Position
    model: chainer.Chain
    gpu: int
    dnn_converter: DNNConverter
    softmax_temperature: float

    def __init__(self):
        self.pos = Position()
        self.model = None
        self.gpu = -1
        self.dnn_converter = DNNConverter(1, 1)
        self.softmax_temperature = 1.0

    @property
    def name(self):
        return "NeneShogi ZeroSearch"

    @property
    def author(self):
        return "select766"

    def get_options(self):
        return {"model_path": "filename default <empty>",
                "softmax_temperature": "string default 1",
                "gpu": "spin default -1 min -1 max 0"}

    def isready(self, options: Dict[str, str]):
        self.softmax_temperature = float(options["softmax_temperature"])
        self.gpu = int(options["gpu"])
        self.model = load_model(options["model_path"])
        if self.gpu >= 0:
            chainer.cuda.get_device_from_id(self.gpu).use()
            self.model.to_gpu()

    def position(self, command: str):
        PositionHelper.set_usi_position(self.pos, command)

    def _make_strategy(self, usi_info_writer: UsiInfoWriter) -> Move:
        """
        方策関数を呼び出して手を決定する
        :return:
        """
        if self.pos.is_mated():
            return Move.MOVE_RESIGN
        dnn_input = self.dnn_converter.get_board_array(self.pos)[np.newaxis, ...]
        legal_move_mask = self.dnn_converter.get_legal_move_array(self.pos)
        with chainer.using_config("train", False):
            if self.gpu >= 0:
                dnn_input = chainer.cuda.to_gpu(dnn_input)
            model_output_var_move, model_output_var_value = self.model.forward(dnn_input)
            model_output = chainer.cuda.to_cpu(model_output_var_move.data)  # type: np.ndarray
        # softmaxで確率とみなす（合法手の和=1）
        mo_exp = np.exp((model_output.flatten() - np.max(model_output)) * self.softmax_temperature) * legal_move_mask.flatten()
        model_output = mo_exp / np.sum(mo_exp)
        # 確率にしたがって手を選択
        move_index = np.random.choice(np.arange(len(model_output)), p=model_output)
        move = self.dnn_converter.reverse_move_index(self.pos, move_index)
        usi_info_writer.write_string(f"{move.to_usi_string()}({int(model_output[move_index]*100)}%)")
        return move

    @util.release_gpu_memory_pool
    def go(self, usi_info_writer: UsiInfoWriter, go_receive_time: float, btime: Optional[int] = None,
           wtime: Optional[int] = None,
           byoyomi: Optional[int] = None, binc: Optional[int] = None, winc: Optional[int] = None):
        move = self._make_strategy(usi_info_writer)
        return move.to_usi_string()
