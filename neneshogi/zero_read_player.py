"""
DNN方策関数による、0手読みプレイヤーの実装
"""
import random
from typing import Dict, Optional, List

import numpy as np
import chainer
import sys

from .position import Position, Color, Square, Piece, Move
from .engine import Engine
from .usi import Usi
from .train_config import load_model


class ZeroReadPlayer(Engine):
    pos: Position

    def __init__(self):
        self.pos = Position()

    @property
    def name(self):
        return "NeneShogi ZeroRead"

    @property
    def author(self):
        return "select766"

    def get_options(self):
        return {"model_path": "filename default <empty>"}

    def isready(self, options: Dict[str, str]):
        model_path = options["model_path"]
        self.model = load_model(model_path)

    def position(self, command: str):
        self.pos.set_usi_position(command)

    def _make_dnn_input(self):
        """
        現在のPositionからDNNへの入力行列を生成する。
        1*61ch*9*9
        DNN入出力の形式は@not522による以下を参考にした。
        https://github.com/not522/CNNShogi
        :return:
        """
        # 常に現在の手番が先手となるように盤面を与える
        pos_from_side = self.pos
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

    def _get_move_index(self, move: Move) -> int:
        """
        行動に対応する行列内indexを取得
        :param move: 現在の手番が手前になるように回転した盤面での手
        :return:
        """
        sq_move_to = move.move_to
        if move.is_drop:
            # 駒打ち
            ch = move.move_dropped_piece - Piece.PAWN + 20
        else:
            # 駒の移動
            sq_move_from = move.move_from
            y_from = Square.file_of(sq_move_from)
            x_from = Square.rank_of(sq_move_from)
            y_to = Square.file_of(sq_move_to)
            x_to = Square.rank_of(sq_move_to)
            if y_to == y_from:
                if x_to < x_from:
                    ch = 0
                else:
                    ch = 1
            elif x_to == y_from:
                if y_to < y_from:
                    ch = 2
                else:
                    ch = 3
            elif (y_to - y_from) == (x_to - x_from):
                if y_to < y_from:
                    ch = 4
                else:
                    ch = 5
            elif (y_to - y_from) == (x_from - x_to):
                if y_to < y_from:
                    ch = 6
                else:
                    ch = 7
            else:
                if y_to < y_from:
                    ch = 8
                else:
                    ch = 9
            if move.is_promote:
                ch += 10
        ary_index = ch * 81 + sq_move_to
        return ary_index

    def _make_sterategy(self, move_list: List[Move]):
        """
        方策関数を呼び出して手を決定する
        :return:
        """
        dnn_input = self._make_dnn_input()
        with chainer.using_config("train", False):
            model_output_var_move, model_output_var_value = self.model.forward(dnn_input)
            model_output = model_output_var_move.data  # type: np.ndarray
        # 表示上、softmaxをかけて確率にしておく
        mo_exp = np.exp(model_output)
        model_output = mo_exp / np.sum(mo_exp)
        if self.pos.side_to_move == Color.BLACK:
            rot_move_list = move_list
        else:
            rot_move_list = []
            for rot_move in move_list:
                to_sq = Square.SQ_NB - 1 - rot_move.move_to
                if rot_move.is_drop:
                    move = Move.make_move_drop(rot_move.move_dropped_piece, to_sq)
                else:
                    from_sq = Square.SQ_NB - 1 - rot_move.move_from
                    move = Move.make_move(from_sq, to_sq, rot_move.is_promote)
                rot_move_list.append(move)
        # 各合法手のスコアを計算し最大値をとる
        max_move = None
        max_score = -1.0
        for move, rot_move in zip(move_list, rot_move_list):
            ary_index = self._get_move_index(rot_move)
            score = model_output[0, ary_index]
            if score > max_score:
                max_score = score
                max_move = move
        # TODO: 読み筋を出力する機能をUSI側に移動
        sys.stdout.write(f"info string {max_move.to_usi_string()}({int(max_score*100)}%)\n")
        return max_move

    def go(self, btime: Optional[int] = None, wtime: Optional[int] = None,
           byoyomi: Optional[int] = None, binc: Optional[int] = None, winc: Optional[int] = None):
        move_list = self.pos.generate_move_list()
        if len(move_list) == 0:
            return "resign"

        move = self._make_sterategy(move_list)
        return move.to_usi_string()

def main():
    import logging

    logger = logging.getLogger("zero_read_player")
    try:
        engine = ZeroReadPlayer()
        logger.debug("Start USI")
        usi = Usi(engine)
        usi.run()
        logger.debug("Quit USI")
    except Exception as ex:
        logger.exception("Unhandled error %s", ex)


if __name__ == "__main__":
    main()
