"""
yaneuraouモジュールのsearchを使ったプレイヤーの実装
"""
import random
import time
from typing import Dict, Optional, Tuple, List
import numpy as np
import threading

from logging import getLogger

logger = getLogger(__name__)

from .move import Piece, Color, Move
from .position import Position, PositionHelper
from .engine import Engine
from .usi_info_writer import UsiInfoWriter


class YsearchPlayer(Engine):
    pos: Position
    depth: int

    def __init__(self):
        self.pos = Position()
        self.depth = 1

    @property
    def name(self):
        return "NeneShogi Ysearch"

    @property
    def author(self):
        return "select766"

    def get_options(self):
        return {"depth": "spin default 3 min 1 max 10"}

    def isready(self, options: Dict[str, str]):
        self.depth = int(options["depth"])

    def position(self, command: str):
        PositionHelper.set_usi_position(self.pos, command)

    def go(self, usi_info_writer: UsiInfoWriter, go_receive_time: float, btime: Optional[int] = None,
           wtime: Optional[int] = None, byoyomi: Optional[int] = None, binc: Optional[int] = None,
           winc: Optional[int] = None):
        best_move = self.pos.search(self.depth)
        return best_move.to_usi_string()
