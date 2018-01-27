"""
盤面を表現するクラス
関数名等は、可能な限りやねうら王に準拠
"""

from .move import Color, Piece, Square, Move
from yaneuraou import Position


class PositionHelper:
    @staticmethod
    def set_usi_position(pos: Position, position_command: str):
        """
        USIの"position"コマンドに従って局面をセットする。
        :param position_command: "position startpos moves 2h6h"
        :return:
        """
        items = position_command.rstrip().split()
        assert items.pop(0) == "position"
        if items[0] == "startpos":
            items.pop(0)
            pos.set_hirate()
        elif items[0] == "sfen":
            items.pop(0)
            # position sfen lnsg... b - 3 moves 7g7f ...
            pos.set_sfen(" ".join(items[:4]))
            del items[:4]
        else:
            raise NotImplementedError
        if len(items) > 0:  # 将棋所で初形だと"position startpos"で終わり
            assert items.pop(0) == "moves"
            for move_str in items:
                move = Move.from_usi(move_str)
                pos.do_move(move)
