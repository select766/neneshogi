"""
DNN評価値関数による、N手読みプレイヤーの実装

反復深化を行い、方策関数によって上位の手だけを読む。

木のノードに付与される情報
- 葉のとき
  - 静的評価値(or予約index)
  - 子ノード展開方策値
  - 詰み局面(展開不能)かどうか
- 内部ノードのとき
  - 指し手と子ノードのリスト

深さN-1の木があるとき、次の操作で深さNの木を構築し指し手を決定する。
1. 木の深さNまでの展開
深さN-1の時点(その時点の葉ノード)で、方策値をもとに上位の合法手を列挙、1段深い木を作成。
深さNに達したら、その局面を表す入力行列を生成し評価リストに登録。評価リスト中のインデックスを保持。
深さNの前に木が展開できなくなったら、詰みなので評価値無限大を保持。
2. 評価
評価リストをバッチサイズごとに区切ってDNNで評価値・次の1手の方策(各指し手の確率)を与える。
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


class ValueProxy:
    """
    DNNの評価値を参照するオブジェクト
    盤面を与えて作成し、後で値をバッチ計算する
    """
    _dnn_input: np.ndarray
    resolved: bool
    value: np.float32
    child_move_probabilities: np.ndarray

    def __init__(self, pos: Position):
        # posは今後変わるものと想定し、ここで内容を読み取っておく
        self._dnn_input = self._make_dnn_input(pos)
        self._resolved = False
        self.value = None
        self.child_move_probabilities = None

    def _make_dnn_input(self, pos: Position):
        """
        PositionからDNNへの入力行列を生成する。
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


class ValueProxyBatch:
    """
    DNNへの入力を蓄積し、バッチサイズだけ蓄積されたら計算を行う処理
    """
    model: chainer.Chain
    gpu: int
    batchsize: int
    items: List[ValueProxy]
    resolve_count: int

    def __init__(self, model: chainer.Chain, gpu: int, batchsize: int):
        self.items = []
        self.model = model
        self.gpu = gpu
        self.batchsize = batchsize
        self.resolve_count = 0

    def append(self, item: ValueProxy):
        self.items.append(item)
        if len(self.items) >= self.batchsize:
            self.resolve()

    def resolve(self):
        """
        蓄積された要素をDNNに投入し解決する
        :return:
        """
        if len(self.items) == 0:
            return
        dnn_input = np.concatenate([item._dnn_input for item in self.items], axis=0)
        if self.gpu >= 0:
            dnn_input = chainer.cuda.to_gpu(dnn_input)
        model_output_var_move, model_output_var_value = self.model.forward(dnn_input)
        model_output_move = chainer.cuda.to_cpu(model_output_var_move.data)
        model_output_value = chainer.cuda.to_cpu(model_output_var_value.data)
        for i, item in enumerate(self.items):
            item.resolved = True
            item.value = model_output_value[i]
            item.child_move_probabilities = model_output_move[i]
        self.resolve_count += len(self.items)
        self.items = []


class GameTreeNode:
    is_leaf: bool
    is_mated: bool
    children: Dict[Move, "GameTreeNode"]
    pv: Move
    value_proxy: ValueProxy

    def __init__(self, pos: Position, value_proxy_batch: ValueProxyBatch):
        self.children = {}
        self.pv = None
        self.is_leaf = True
        self.is_mated = False
        self.value_proxy = ValueProxy(pos)
        value_proxy_batch.append(self.value_proxy)

    def get_value(self) -> float:
        if self.is_mated:
            # 詰んでいる
            # 手番側からみた値なので、大きな負の値
            return -50.0
        if self.is_leaf:
            assert self.value_proxy.resolved
            return self.value_proxy.value
        else:
            # 子の評価値を反転して最大をとる(negamax)
            max_move = None
            max_value = -100.0
            for move, node in self.children.items():
                node_value = -node.get_value()
                if node_value > max_value:
                    max_value = node_value
                    max_move = move
            self.pv = max_move
            return max_value

    def get_pv(self) -> List[Move]:
        """
        読み筋を出力。get_valueの後に呼び出し可能となる。
        :return:
        """
        if self.pv is None:
            return []
        else:
            return [self.pv] + self.children[self.pv].get_pv()

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

    def expand_child(self, pos: Position, value_proxy_batch: ValueProxyBatch):
        """
        現在葉ノードである場合に、実現確率の高い子ノードを作成する。
        :param pos: このノードに対応するPosition
        :return:
        """
        if self.is_mated:
            # 詰んでいることがすでにわかっている
            return
        move_list = pos.generate_move_list()
        if len(move_list) == 0:
            # 詰んでいるので展開できない
            self.is_mated = True
            return

        # rot_move_list: DNNの出力インデックス計算のため、手番側が手前になるように回転したmove
        if pos.side_to_move == Color.BLACK:
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

        score_moves = []
        for i in range(len(move_list)):
            move_index = self._get_move_index(rot_move_list[i])
            move_prob = self.value_proxy.child_move_probabilities[move_index]
            score_moves.append((-move_prob, i, move_list[i]))  # iを入れることで、moveの比較が起こらないようにする
        score_moves.sort()  # move_probが大きい順に並び替え

        # TODO: top Nの値を設定
        top_moves = [item[2] for item in score_moves[:3]]  # type: List[Move]
        for move in top_moves:
            undo_info = pos.do_move(move)
            child_node = GameTreeNode(pos, value_proxy_batch)
            self.children[move] = child_node
            pos.undo_move(undo_info)
        self.is_leaf = False
        self.value_proxy = None  # もう静的評価値・方策は使わないので解放


class NarrowSearchPlayer(Engine):
    pos: Position
    model: chainer.Chain
    depth: int
    batchsize: int
    gpu: int
    value_proxy_batch: ValueProxyBatch
    nodes_count: int  # ある局面の探索開始からのノード数

    def __init__(self):
        self.pos = Position()
        self.model = None
        self.gpu = -1
        self.depth = 1
        self.batchsize = 256
        self.value_proxy_batch = None
        self.nodes_count = 0

    @property
    def name(self):
        return "NeneShogi NarrowSearch"

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
        self.value_proxy_batch = ValueProxyBatch(self.model, self.gpu, self.batchsize)

    def position(self, command: str):
        self.pos.set_usi_position(command)

    def do_search_recursion(self, node: GameTreeNode):
        if node.is_leaf:
            # 葉なら、1つ深く展開
            node.expand_child(self.pos, self.value_proxy_batch)
        else:
            # 内部ノードなら、1つ深いノードで再帰的に計算
            for move, child_node in node.children.items():
                undo_info = self.pos.do_move(move)
                self.do_search_recursion(child_node)
                self.pos.undo_move(undo_info)

    def do_search_root(self, tree_root: GameTreeNode, depth: int):
        """
        深さ"depth"まで読んで木を更新する
        :return:
        """
        logger.info(f"generating game tree of depth {depth}")
        self.value_proxy_batch.resolve_count = 0
        # 木を掘り下げる
        self.do_search_recursion(tree_root)
        # 評価値計算
        self.value_proxy_batch.resolve()
        self.nodes_count += self.value_proxy_batch.resolve_count
        logger.info(f"Calculated {self.value_proxy_batch.resolve_count} positions")
        # 読み筋を計算
        root_value = tree_root.get_value()
        pv = tree_root.get_pv()
        logger.info("done")
        if len(pv) == 0:
            return "resign"

        pv_str = " ".join([move.to_usi_string() for move in pv])
        sys.stdout.write(f"info depth {depth} nodes {self.nodes_count} score cp {int(root_value * 600)} pv {pv_str}\n")
        sys.stdout.flush()
        return pv[0].to_usi_string()

    def generate_tree_root(self) -> GameTreeNode:
        """
        ルートノードを作成し、方策を計算する
        :return:
        """
        tree_root = GameTreeNode(self.pos, self.value_proxy_batch)
        self.value_proxy_batch.resolve()
        return tree_root

    def go(self, btime: Optional[int] = None, wtime: Optional[int] = None,
           byoyomi: Optional[int] = None, binc: Optional[int] = None, winc: Optional[int] = None):
        self.nodes_count = 0
        tree_root = self.generate_tree_root()

        move_str = "resign"
        for cur_depth in range(1, self.depth + 1):
            move_str = self.do_search_root(tree_root, cur_depth)
        return move_str


def main():
    try:
        engine = NarrowSearchPlayer()
        logger.debug("Start USI")
        usi = Usi(engine)
        usi.run()
        logger.debug("Quit USI")
    except Exception as ex:
        logger.exception("Unhandled error %s", ex)


if __name__ == "__main__":
    import logging

    logger = logging.getLogger("narrow_search_player")
    main()
