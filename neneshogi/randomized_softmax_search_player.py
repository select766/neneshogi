"""
DNN評価値関数による、Randomized Softmax Search Playerの実装
原型は「芝浦将棋Softmax」(WCSC27)のアルゴリズム。
GPUでバッチ処理するための変更を加えている。

ゲーム木上で方策関数により、実現確率に比例した確率で手を選ぶ。
未評価(評価値がない)の局面に達したら、その評価値・方策を計算。
最終的にできたゲーム木をminimax法で探索して手を決定。
"""
import random
from typing import Dict, Optional, List, Tuple

from logging import getLogger

logger = getLogger(__name__)

import numpy as np
import chainer

from .position import Position, Color, Square, Piece, Move
from .engine import Engine
from .usi_info_writer import UsiInfoWriter
from .train_config import load_model
from . import util


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
    softmax_temperature: float
    unique_pos_set: set

    def __init__(self, model: chainer.Chain, gpu: int, batchsize: int):
        self.items = []
        self.model = model
        self.gpu = gpu
        self.batchsize = batchsize
        self.resolve_count = 0
        self.softmax_temperature = 1.0
        self.unique_pos_set = set()

    def append(self, item: ValueProxy):
        self.unique_pos_set.add(hash(item._dnn_input.tobytes()))
        self.items.append(item)
        if len(self.items) >= self.batchsize:
            self.resolve()

    def resolve(self):
        """
        蓄積された要素をDNNに投入し解決する
        :return:
        """
        logger.info(f"evaluating {len(self.items)} positions")
        if len(self.items) == 0:
            return
        with chainer.using_config('train', False):
            with chainer.using_config('enable_backprop', False):
                dnn_input = np.concatenate([item._dnn_input for item in self.items], axis=0)
                if self.gpu >= 0:
                    dnn_input = chainer.cuda.to_gpu(dnn_input)
                model_output_var_move, model_output_var_value = self.model.forward(dnn_input)
                model_output_var_move_softmax = chainer.functions.softmax(
                    model_output_var_move * (1.0 / self.softmax_temperature))
                model_output_move = chainer.cuda.to_cpu(model_output_var_move_softmax.data)
                model_output_value = chainer.cuda.to_cpu(model_output_var_value.data)
                # 進行をばらけさせるために評価値をすこしランダムにずらす
                # TODO: ずらす幅の検証
                model_output_value += \
                    np.random.normal(loc=0.0, scale=0.01, size=model_output_value.shape) \
                        .astype(model_output_value.dtype)
        for i, item in enumerate(self.items):
            item.resolved = True
            item.value = model_output_value[i]
            item.child_move_probabilities = model_output_move[i]
        self.resolve_count += len(self.items)
        self.items = []


class GameTreeNode:
    legal_moves: List[Move]
    parent: Optional["GameTreeNode"]
    move_from_parent: Optional[Move]
    is_leaf: bool
    children: Dict[Move, "GameTreeNode"]
    pv: Move
    value_proxy: ValueProxy
    value_parsed: bool  # value_proxyの結果を取り出したかどうか
    static_value: float  # 静的評価値
    legal_move_probabilities: np.ndarray  # 合法手それぞれの実現確率(legal_movesの順序と対応)

    def __init__(self, pos: Position, parent: Optional["GameTreeNode"], move_from_parent: Optional[Move],
                 value_proxy_batch: ValueProxyBatch):
        self.legal_moves = pos.generate_move_list()
        self.is_leaf = True
        self.parent = parent
        self.move_from_parent = move_from_parent
        self.children = {}
        self.pv = None
        self.legal_move_probabilities = None
        if not self.is_mated:
            # 詰みの時は評価値不要
            self.value_proxy = ValueProxy(pos)
            value_proxy_batch.append(self.value_proxy)
            self.value_parsed = False
        else:
            # 詰んでいる
            # 手番側からみた値なので、大きな負の値
            self.static_value = -50.0
            self.value_parsed = True

    @property
    def is_mated(self):
        return len(self.legal_moves) == 0

    def add_child(self, move: Move, child_node: "GameTreeNode") -> None:
        self.children[move] = child_node
        self.is_leaf = False

    def get_value(self) -> float:
        if self.is_mated:
            return self.static_value
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

    def parse_value_if_needed(self, pos: Position):
        """
        計算済みの評価値を取り出してstatic_value, legal_move_probabilitiesに格納
        :param pos:
        :return:
        """
        if self.value_parsed:
            return
        assert self.value_proxy.resolved

        self.static_value = float(self.value_proxy.value)
        # rot_move_list: DNNの出力インデックス計算のため、手番側が手前になるように回転したmove
        move_list = self.legal_moves
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

        prob_moves = np.zeros((len(move_list),), dtype=np.float32)
        for i in range(len(move_list)):
            move_index = self._get_move_index(rot_move_list[i])
            move_prob = self.value_proxy.child_move_probabilities[move_index]
            prob_moves[i] = move_prob
        # np.random.choiceの制約で、確率の和が1である必要あり
        prob_moves /= np.sum(prob_moves)
        self.legal_move_probabilities = prob_moves

        self.value_proxy = None  # もう静的評価値・方策は使わないので解放
        self.value_parsed = True

    def sample_child(self, pos: Position, value_proxy_batch: ValueProxyBatch) -> Optional["GameTreeNode"]:
        """
        実現確率に従って再帰的に末端ノードへ行き、新規ノードを生成する。
        :param pos: このノードに対応するPosition
        :return: 新規作成されたノード(詰みなどでNoneの場合あり)
        """
        if self.is_mated:
            return None
        self.parse_value_if_needed(pos)
        chosen_move_index = np.random.choice(self.legal_move_probabilities.size, p=self.legal_move_probabilities)
        chosen_move = self.legal_moves[chosen_move_index]
        undo_info = pos.do_move(chosen_move)
        if chosen_move in self.children:
            new_child = self.children[chosen_move].sample_child(pos, value_proxy_batch)
        else:
            # 新規ノード(仮ノード)の作成
            new_child = GameTreeNode(pos, self, chosen_move, value_proxy_batch)
            # 現時点では子ノードとして追加せず、評価値計算が終わってから追加する
        pos.undo_move(undo_info)
        return new_child

    def expand_all_children(self, pos: Position, value_proxy_batch: ValueProxyBatch) -> List["GameTreeNode"]:
        """
        すべての合法手に対応する子ノードを一括作成する。
        :param pos:
        :param value_proxy_batch:
        :return:
        """
        assert not self.is_mated, "No legal moves"
        children = []
        for move in self.legal_moves:
            undo_info = pos.do_move(move)
            new_child = GameTreeNode(pos, self, move, value_proxy_batch)
            children.append(new_child)
            pos.undo_move(undo_info)
        return children


class RandomizedSoftmaxSearchPlayer(Engine):
    pos: Position
    model: chainer.Chain
    depth: int
    batchsize: int
    gpu: int
    softmax_temperature: float
    value_proxy_batch: ValueProxyBatch
    nodes_count: int  # ある局面の探索開始からのノード数
    tmp_nodes: List[GameTreeNode]

    def __init__(self):
        self.pos = Position()
        self.model = None
        self.gpu = -1
        self.depth = 1
        self.batchsize = 256
        self.value_proxy_batch = None
        self.nodes_count = 0
        self.tmp_nodes = []
        self.softmax_temperature = 1.0

    @property
    def name(self):
        return "NeneShogi RandomizedSoftmaxSearch"

    @property
    def author(self):
        return "select766"

    def get_options(self):
        return {"model_path": "filename default <empty>",
                "gpu": "spin default -1 min -1 max 0",
                "depth": "spin default 1 min 1 max 5",
                "softmax_temperature": "string default 1"}

    def isready(self, options: Dict[str, str]):
        self.gpu = int(options["gpu"])
        self.depth = int(options["depth"])
        self.model = load_model(options["model_path"])
        self.softmax_temperature = float(options["softmax_temperature"])
        if self.gpu >= 0:
            chainer.cuda.get_device_from_id(self.gpu).use()
            self.model.to_gpu()
        self.value_proxy_batch = ValueProxyBatch(self.model, self.gpu, self.batchsize)
        self.value_proxy_batch.softmax_temperature = self.softmax_temperature
        # TODO: ここで一度NNを走らせて、CUDAカーネルの初期化をさせたほうがよい

    def position(self, command: str):
        self.pos.set_usi_position(command)
        logger.info(f"Position set to {self.pos.get_sfen()}")

    def do_search_root(self, usi_info_writer: UsiInfoWriter, tree_root: GameTreeNode, iter_index: int):
        """
        バッチサイズ分探索をして、指し手を決める
        :return:
        """
        logger.info(f"generating game tree, iteration {iter_index}")
        self.value_proxy_batch.resolve_count = 0
        # 探索で木を広げる
        while len(self.tmp_nodes) < self.batchsize:
            new_node = tree_root.sample_child(self.pos, self.value_proxy_batch)
            if new_node is not None:
                self.tmp_nodes.append(new_node)
        # 各ノードに評価値を与える
        self.resolve_and_attach()
        self.nodes_count += self.value_proxy_batch.resolve_count
        logger.info(f"Calculated {self.value_proxy_batch.resolve_count} positions")
        # 読み筋を計算
        root_value = tree_root.get_value()
        pv = tree_root.get_pv()
        logger.info("done")
        if len(pv) == 0:
            return "resign"

        usi_info_writer.write_pv(pv=pv, depth=int(iter_index), nodes=self.nodes_count, score_cp=int(root_value * 600))
        # PV沿いの確率を表示
        pv_path = tree_root
        pv_items = []
        for pv_move in pv:
            idx = pv_path.legal_moves.index(pv_move)
            prob = pv_path.legal_move_probabilities[idx]
            max_prob = np.max(pv_path.legal_move_probabilities)
            pv_items.append(pv_move.to_usi_string())
            pv_items.append(f"{int(prob*100)}%/{int(max_prob*100)}%")
            pv_path = pv_path.children[pv_move]
        usi_info_writer.write_string(" ".join(pv_items))
        usi_info_writer.write_string(f"Unique pos: {len(self.value_proxy_batch.unique_pos_set)}")
        return pv[0].to_usi_string()

    def generate_initial_tree(self) -> GameTreeNode:
        """
        初期ゲーム木を作成する。
        ルート局面およびその先の1手すべてを含む。
        :return:
        """
        tree_root = GameTreeNode(self.pos, None, None, self.value_proxy_batch)
        self.tmp_nodes.append(tree_root)
        #if not tree_root.is_mated:
        #    self.tmp_nodes.extend(tree_root.expand_all_children(self.pos, self.value_proxy_batch))

        return tree_root

    def resolve_and_attach(self):
        """
        仮ノードの評価を完了させ、ゲーム木に追加する
        :return:
        """
        self.value_proxy_batch.resolve()
        for tmp_node in self.tmp_nodes:
            if tmp_node.parent is not None:  # ルートノードでは親がない
                tmp_node.parent.add_child(tmp_node.move_from_parent, tmp_node)
        self.tmp_nodes.clear()

    @util.release_gpu_memory_pool
    def go(self, usi_info_writer: UsiInfoWriter, btime: Optional[int] = None, wtime: Optional[int] = None,
           byoyomi: Optional[int] = None, binc: Optional[int] = None, winc: Optional[int] = None):
        self.nodes_count = 0
        self.value_proxy_batch.unique_pos_set.clear()
        tree_root = self.generate_initial_tree()
        self.resolve_and_attach()

        if tree_root.is_mated:
            return "resign"
        move_str = "resign"
        for iter_index in range(self.depth):
            move_str = self.do_search_root(usi_info_writer, tree_root, iter_index)
        return move_str
