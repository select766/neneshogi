"""
ランダムプレイヤーの実装
"""
import random
from typing import Dict, Optional

from .position import Position
from .engine import Engine
from .usi import Usi


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
        self.pos.set_usi_position(command)

    def go(self, btime: Optional[int] = None, wtime: Optional[int] = None,
           byoyomi: Optional[int] = None, binc: Optional[int] = None, winc: Optional[int] = None):
        move_list = self.pos.generate_move_list()
        if len(move_list) == 0:
            return "resign"

        move = random.choice(move_list)
        return move.to_usi_string()


if __name__ == "__main__":
    import logging
    logger = logging.getLogger("random_player")
    try:
        engine = RandomPlayer()
        logger.debug("Start USI")
        usi = Usi(engine)
        usi.run()
        logger.debug("Quit USI")
    except Exception as ex:
        logger.exception("Unhandled error %s", ex)
