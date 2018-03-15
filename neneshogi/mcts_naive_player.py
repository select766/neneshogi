"""
DNN評価値関数による、MCTSプレイヤーの実装
ベースラインtのして、1サンプルずつ評価する非常にナイーブな実装
"""
import random
from typing import Dict, Optional, List, Tuple

from logging import getLogger

logger = getLogger(__name__)

import numpy as np
import scipy.special
import chainer
import chainer.functions as F

from .position import Position, Color, Square, Piece, Move, PositionHelper
from yaneuraou import DNNConverter
from .engine import Engine
from .usi_info_writer import UsiInfoWriter
from .train_config import load_model
from . import util


class TreeConfig:
    c_puct: float
    play_temperature: float

    def __init__(self):
        self.c_puct = 1.0
        self.play_temperature = 1.0


class TreeSelectResult:
    final_node: "TreeNode"
    final_edge_index: int
    moves: List[Move]

    def __init__(self):
        pass


class TreeNode:
    tree_config: TreeConfig
    parent: "TreeNode"
    parent_edge_index: int
    move_list: List[Move]
    score: float
    children: List["TreeNode"]
    value_n: np.ndarray
    value_w: np.ndarray
    value_q: np.ndarray
    value_p: np.ndarray
    terminal: bool

    def __init__(self, tree_config: TreeConfig, parent: "TreeNode", parent_edge_index: int, move_list: List[Move],
                 score: float, value_p: np.ndarray):
        self.tree_config = tree_config
        self.parent = parent
        self.parent_edge_index = parent_edge_index
        self.move_list = move_list
        self.score = score

        n_children = len(move_list)
        if n_children == 0:
            self.terminal = True
            return
        self.terminal = False
        self.value_p = value_p
        self.children = [None] * n_children
        self.value_n = np.zeros((n_children,), dtype=np.float32)
        self.value_w = np.zeros((n_children,), dtype=np.float32)
        self.value_q = np.zeros((n_children,), dtype=np.float32)
        self._backup()

    def _backup(self):
        cur_edge = self.parent_edge_index
        cur_node = self.parent
        score = -self.score  # このノードの評価値が高ければ、相手手番であるself.parentからは選ばれにくくなるのが正しい
        while cur_node is not None:
            cur_node.value_n[cur_edge] += 1
            cur_node.value_w[cur_edge] += score
            cur_node.value_q[cur_edge] = cur_node.value_w[cur_edge] / cur_node.value_n[cur_edge]
            cur_edge = cur_node.parent_edge_index
            cur_node = cur_node.parent
            score = -score

    def _select_edge(self) -> int:
        assert not self.terminal
        n_sum_sqrt = np.sqrt(np.sum(self.value_n) + 0.01)
        value_u = self.value_p / (self.value_n + 1) * (self.tree_config.c_puct * n_sum_sqrt)
        best = np.argmax(self.value_q + value_u)
        return int(best)

    def select(self) -> Optional[TreeSelectResult]:
        if self.terminal:
            # 詰みノード
            # 評価は不要で、親へ評価値を再度伝播する
            logger.info("selected terminal node")
            self._backup()
            return None
        edge = self._select_edge()
        child = self.children[edge]
        if child is None:
            # 子ノードがまだ生成されていない
            tsr = TreeSelectResult()
            tsr.final_edge_index = edge
            tsr.final_node = self
            tsr.moves = [self.move_list[edge]]
        else:
            tsr = child.select()
            if tsr is None:
                return None
            tsr.moves.insert(0, self.move_list[edge])
        return tsr

    def play(self) -> Tuple[Move, float]:
        assert not self.terminal
        # logsumexpを使ってオーバーフロー回避
        # value_n_exp = np.power(self.value_n, (1.0 / self.tree_config.play_temperature))
        # probs = value_n_exp / np.sum(value_n_exp)
        temp_log_value_n = (1.0 / self.tree_config.play_temperature) * np.log(self.value_n + 1e-20)
        denom = scipy.special.logsumexp(temp_log_value_n)  # type: np.ndarray
        probs = np.exp(temp_log_value_n - denom)
        logger.info("Probs: {}".format([(self.move_list[i], probs[i]) for i in np.argsort(-probs)]))
        selected_edge = np.random.choice(np.arange(len(probs)), p=probs)
        return self.move_list[selected_edge], probs[selected_edge]


class MCTSNaivePlayer(Engine):
    pos: Position
    model: chainer.Chain
    gpu: int
    nodes: int
    dnn_converter: DNNConverter
    tree_config: TreeConfig

    def __init__(self):
        self.pos = Position()
        self.model = None
        self.gpu = -1
        self.dnn_converter = DNNConverter(1, 1)

    @property
    def name(self):
        return "NeneShogi MCTSNaive"

    @property
    def author(self):
        return "select766"

    def get_options(self):
        return {"model_path": "filename default <empty>",
                "nodes": "string default 100",
                "c_puct": "string default 1",
                "play_temperature": "string default 1",
                "gpu": "spin default -1 min -1 max 0"}

    def isready(self, options: Dict[str, str]):
        self.nodes = int(options["nodes"])
        self.tree_config = TreeConfig()
        self.tree_config.c_puct = float(options["c_puct"])
        self.tree_config.play_temperature = float(options["play_temperature"])
        self.gpu = int(options["gpu"])
        self.model = load_model(options["model_path"])
        if self.gpu >= 0:
            chainer.cuda.get_device_from_id(self.gpu).use()
            self.model.to_gpu()

    def position(self, command: str):
        PositionHelper.set_usi_position(self.pos, command)

    def _eval_current_pos(self, parent: "TreeNode", parent_edge_index: int) -> TreeNode:
        move_list = self.pos.generate_move_list()
        if len(move_list) == 0:
            # mated
            return TreeNode(self.tree_config, parent, parent_edge_index, move_list, -1.0, None)
        dnn_input = self.dnn_converter.get_board_array(self.pos)[np.newaxis, ...]
        with chainer.using_config("train", False):
            if self.gpu >= 0:
                dnn_input = chainer.cuda.to_gpu(dnn_input)
            model_output_var_move, model_output_var_value = self.model.forward(dnn_input)
            model_output_var_move = F.softmax(model_output_var_move)
            model_output_var_value = F.tanh(model_output_var_value)
            model_output_var_move = chainer.cuda.to_cpu(model_output_var_move.data)
            model_output_var_value = chainer.cuda.to_cpu(model_output_var_value.data)
        value_p = []
        for move in move_list:
            value_p.append(model_output_var_move[0, self.dnn_converter.get_move_index(self.pos, move)])
        return TreeNode(self.tree_config, parent, parent_edge_index, move_list,
                        float(model_output_var_value[0]), np.array(value_p, dtype=np.float32))

    def _make_strategy(self, usi_info_writer: UsiInfoWriter):
        """
        1手展開した結果に対し、評価関数を呼び出して手を決定する
        :return:
        """
        if self.pos.is_mated():
            return Move.MOVE_RESIGN

        root_node = self._eval_current_pos(None, 0)
        for i in range(self.nodes):
            tsr = root_node.select()
            if tsr is None:
                continue
            for move in tsr.moves:
                self.pos.do_move(move)
            new_node = self._eval_current_pos(tsr.final_node, tsr.final_edge_index)
            for j in range(len(tsr.moves)):
                self.pos.undo_move()
            tsr.final_node.children[tsr.final_edge_index] = new_node
        best_move, prob = root_node.play()
        usi_info_writer.write_string(f"{best_move.to_usi_string()}({int(prob*100)}%)")
        return best_move

    @util.release_gpu_memory_pool
    def go(self, usi_info_writer: UsiInfoWriter, go_receive_time: float, btime: Optional[int] = None,
           wtime: Optional[int] = None, byoyomi: Optional[int] = None, binc: Optional[int] = None,
           winc: Optional[int] = None):
        move = self._make_strategy(usi_info_writer)
        return move.to_usi_string()
