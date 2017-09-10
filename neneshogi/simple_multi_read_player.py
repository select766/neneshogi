"""
DNN評価値関数による、N手読みプレイヤーの実装

単純にゲーム木を深さNまですべて展開し、minimax法で指し手を決める

1. 木の深さNまでの展開
深さNに達したら、その局面を表す入力行列を生成し評価リストに登録。評価リスト中のインデックスを保持。
深さNの前に木が展開できなくなったら、詰みなので評価値無限大を保持。
2. 評価
評価リストをバッチサイズごとに区切ってDNNで評価値を与える。
3. minimax法による木の評価
末端局面では、局面に対する評価値が判明しているのでそれを返す。
途中の局面では、最大評価値の子ノードを見つけてその値と該当子ノード(読み筋)を記録。
最後にルートノードから読み筋をたどって出力。
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


class GameTreeNode:
    children: Dict[Move, "GameTreeNode"]
    pv: Move
    value: Optional[float]
    value_array_index: Optional[int]

    def __init__(self):
        self.children = {}
        self.pv = None
        self.value = None
        self.value_array_index = None

    def get_value(self, value_array: np.ndarray) -> float:
        if self.value_array_index is not None:
            self.value = value_array[self.value_array_index]
        if self.value is not None:
            return self.value
        assert len(self.children) > 0, "GameTreeNode has neither static value or children"
        # 子の評価値を反転して最大をとる(negamax)
        max_move = None
        max_value = -100.0
        for move, node in self.children.items():
            node_value = -node.get_value(value_array)
            if node_value > max_value:
                max_value = node_value
                max_move = move
        self.pv = max_move
        self.value = max_value
        return self.value

    def get_pv(self) -> List[Move]:
        """
        読み筋を出力。get_valueの後に呼び出し可能となる。
        :return:
        """
        if self.pv is None:
            return []
        else:
            return [self.pv] + self.children[self.pv].get_pv()


class SimpleMultiReadPlayer(Engine):
    pos: Position
    model: chainer.Chain
    depth: int
    batchsize: int
    gpu: int
    evaluate_list: List[np.ndarray]

    def __init__(self):
        self.pos = Position()
        self.model = None
        self.gpu = -1
        self.depth = 1
        self.batchsize = 256
        self.evaluate_list = None

    @property
    def name(self):
        return "NeneShogi SimpleMultiRead"

    @property
    def author(self):
        return "select766"

    def get_options(self):
        return {"model_path": "filename default <empty>",
                "gpu": "spin default -1 min -1 max 0",
                "depth": "spin default 1 min 1 max 5"}

    def isready(self, options: Dict[str, str]):
        self.gpu = int(options["gpu"])
        self.depth = int(options["depth"])
        self.model = load_model(options["model_path"])
        if self.gpu >= 0:
            chainer.cuda.get_device_from_id(self.gpu).use()
            self.model.to_gpu()

    def position(self, command: str):
        self.pos.set_usi_position(command)

    def _make_dnn_input(self):
        """
        現在のPositionからDNNへの入力行列を生成する。
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

    def generate_tree(self, current_node: GameTreeNode, depth: int):
        if depth <= 0:
            # 末端局面なので静的評価を予約する
            current_node.value_array_index = len(self.evaluate_list)
            self.evaluate_list.append(self._make_dnn_input())
        else:
            # 1手深く木を作る
            move_list = self.pos.generate_move_list()
            if len(move_list) == 0:
                # 詰んでいる
                # 手番側からみた値なので、大きな負の値
                current_node.value = -50.0
            else:
                for move in move_list:
                    undo_info = self.pos.do_move(move)
                    child_node = GameTreeNode()
                    self.generate_tree(child_node, depth - 1)
                    current_node.children[move] = child_node
                    self.pos.undo_move(undo_info)

    def evaluate_leaf(self) -> np.ndarray:
        model_outputs = []
        with chainer.using_config("train", False):
            for i in range(0, len(self.evaluate_list), self.batchsize):
                dnn_input = np.concatenate(self.evaluate_list[i:i + self.batchsize], axis=0)
                if self.gpu >= 0:
                    dnn_input = chainer.cuda.to_gpu(dnn_input)
                model_output_var_move, model_output_var_value = self.model.forward(dnn_input)
                model_output = chainer.cuda.to_cpu(model_output_var_value.data)  # type: np.ndarray
                model_outputs.append(model_output)
        if len(model_outputs) > 0:
            return np.concatenate(model_outputs)
        else:
            return np.array([], dtype=np.float32)

    def go(self, btime: Optional[int] = None, wtime: Optional[int] = None,
           byoyomi: Optional[int] = None, binc: Optional[int] = None, winc: Optional[int] = None):
        # 木の作成
        logger.info("generating game tree")
        self.evaluate_list = []
        tree_root = GameTreeNode()
        self.generate_tree(tree_root, self.depth)

        # 葉ごとの評価値計算
        logger.info(f"evaluating each leaf ({len(self.evaluate_list)})")
        value_array = self.evaluate_leaf()
        self.evaluate_list = None  # メモリ開放

        # 読み筋の計算
        logger.info("selecting best move")
        root_value = tree_root.get_value(value_array)
        pv = tree_root.get_pv()
        logger.info("done")
        if len(pv) == 0:
            return "resign"

        pv_str = " ".join([move.to_usi_string() for move in pv])
        sys.stdout.write(f"info depth 1 score cp {int(root_value * 600)} pv {pv_str}\n")
        return pv[0].to_usi_string()


def main():
    try:
        engine = SimpleMultiReadPlayer()
        logger.debug("Start USI")
        usi = Usi(engine)
        usi.run()
        logger.debug("Quit USI")
    except Exception as ex:
        logger.exception("Unhandled error %s", ex)


if __name__ == "__main__":
    import logging

    logger = logging.getLogger("simple_multi_read_player")
    main()
