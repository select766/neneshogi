"""
ランダムプレイヤーの実装
"""
import random
from typing import Dict, Optional

from logging import getLogger

logger = getLogger(__name__)

from .position import Position, PositionHelper
from .engine import Engine
from .usi_info_writer import UsiInfoWriter


class RandomPlayer(Engine):
    pos: Position

    def __init__(self):
        self.pos = Position()

    @property
    def name(self):
        return "NeneShogi Random"

    @property
    def author(self):
        return "select766"

    def get_options(self):
        return {}

    def isready(self, options: Dict[str, str]):
        pass

    def position(self, command: str):
        PositionHelper.set_usi_position(self.pos, command)

    def go(self, usi_info_writer: UsiInfoWriter, go_receive_time: float, btime: Optional[int] = None,
           wtime: Optional[int] = None, byoyomi: Optional[int] = None, binc: Optional[int] = None,
           winc: Optional[int] = None):
        move_list = self.pos.generate_move_list()
        if len(move_list) == 0:
            return "resign"

        move = random.choice(move_list)
        return move.to_usi_string()
