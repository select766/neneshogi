"""
DNN評価値関数による、MCTSプレイヤーの実装
局面評価をバッチにまとめて、別プロセスで処理
"""
import random
from typing import Dict, Optional, List, Tuple
import queue
import multiprocessing

from logging import getLogger

import time

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
from .mcts_evaluator import EvalItem, ResultItem, EvaluatorConfig, EvaluatorBase, EvaluatorSingleGPU
from . import util


class EvalWaitItem:
    """
    評価結果を用いてノード作成に必要な情報
    """
    parent: "TreeNode"
    parent_edge_index: int
    move_list: List[Move]
    move_indices: List[int]


class TreeConfig:
    c_puct: float
    play_temperature: float
    virtual_loss: float

    def __init__(self):
        self.c_puct = 1.0
        self.play_temperature = 1.0
        self.virtual_loss = 1.0


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
    virtual_loss_ctr: np.ndarray
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
            self.children = None
            return
        self.terminal = False
        self.value_p = value_p
        self.children = [None] * n_children
        self.value_n = np.zeros((n_children,), dtype=np.float32)
        self.value_w = np.zeros((n_children,), dtype=np.float32)
        self.value_q = np.zeros((n_children,), dtype=np.float32)
        self.virtual_loss_ctr = np.zeros((n_children,), dtype=np.int32)
        self._backup()

    def _backup(self):
        cur_edge = self.parent_edge_index
        cur_node = self.parent
        score = -self.score  # このノードの評価値が高ければ、相手手番であるself.parentからは選ばれにくくなるのが正しい
        while cur_node is not None:
            cur_node.value_n[cur_edge] += 1 - self.tree_config.virtual_loss
            cur_node.value_w[cur_edge] += score + self.tree_config.virtual_loss
            cur_node.value_q[cur_edge] = cur_node.value_w[cur_edge] / cur_node.value_n[cur_edge]
            cur_node.virtual_loss_ctr[cur_edge] -= 1
            cur_edge = cur_node.parent_edge_index
            cur_node = cur_node.parent
            score = -score

    def _restore_virtual_loss(self, edge_index: int):
        self.value_n[edge_index] -= self.tree_config.virtual_loss
        self.value_w[edge_index] += self.tree_config.virtual_loss
        self.value_q[edge_index] = self.value_w[edge_index] / self.value_n[edge_index]
        self.virtual_loss_ctr[edge_index] -= 1
        if self.parent is not None:
            self.parent._restore_virtual_loss(self.parent_edge_index)

    def _select_edge(self) -> int:
        assert not self.terminal
        n_sum_sqrt = np.sqrt(np.sum(self.value_n))
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

        # virtual loss加算
        self.value_n[edge] += self.tree_config.virtual_loss
        self.value_w[edge] -= self.tree_config.virtual_loss
        self.value_q[edge] = self.value_w[edge] / self.value_n[edge]
        self.virtual_loss_ctr[edge] += 1
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

    def _depth_stat_inner(self, cur_depth: int, buf: np.ndarray):
        """
        深さごとのノード数を調べる
        :return:
        """
        buf[cur_depth] += 1
        if self.children is None:
            return
        for child in self.children:
            if child is not None:
                child._depth_stat_inner(cur_depth + 1, buf)

    def depth_stat(self):
        """
        深さごとのノード数を調べる
        :return:
        """
        buf = np.zeros((100,), dtype=np.int32)
        self._depth_stat_inner(0, buf)
        max_depth = np.flatnonzero(buf)[-1]
        logger.info(f"Depth max={max_depth}, hist={buf[:max_depth+1]}")


class MCTSPlayer(Engine):
    pos: Position
    nodes: int
    batch_size: int
    dnn_converter: DNNConverter
    tree_config: TreeConfig
    evaluator: EvaluatorBase
    kifu_gen: bool

    def __init__(self, evaluator: EvaluatorBase = None, kifu_gen: bool = False):
        """
        MCTSPlayer Engine初期化
        :param evaluator: NN Evaluatorのインスタンス
        :param kifu_gen: 棋譜生成モード
        """
        self.pos = Position()
        self.kifu_gen = kifu_gen
        self.model = None
        self.gpu = -1
        self.dnn_converter = DNNConverter(1, 1)
        if evaluator is None:
            evaluator = EvaluatorSingleGPU()
        evaluator.start()
        self.evaluator = evaluator

    @property
    def name(self):
        return "NeneShogi MCTS"

    @property
    def author(self):
        return "select766"

    def get_options(self):
        return {"model_path": "filename default <empty>",
                "nodes": "string default 100",
                "c_puct": "string default 1",
                "play_temperature": "string default 1",
                "batch_size": "string default 32",
                "gpu": "spin default -1 min -1 max 0"}

    def isready(self, options: Dict[str, str]):
        self.nodes = int(options["nodes"])
        self.tree_config = TreeConfig()
        self.tree_config.c_puct = float(options["c_puct"])
        self.tree_config.play_temperature = float(options["play_temperature"])
        self.evaluator.set_batch_size(int(options["batch_size"]))
        if isinstance(self.evaluator, EvaluatorSingleGPU):
            logger.info("Waiting evaluator to initialize")
            eval_config = EvaluatorConfig()
            eval_config.gpu = int(options["gpu"])
            eval_config.model_path = options["model_path"]
            self.evaluator.set_config(eval_config)
        logger.info("End of isready")

    def position(self, command: str):
        PositionHelper.set_usi_position(self.pos, command)

    def _search_once(self, root_node: TreeNode) -> bool:
        """
        ゲーム木を1回たどり、（詰み局面でなければ）新規末端ノードの評価の予約を行う。
        :return:
        """
        tsr = root_node.select()
        if tsr is None:
            # 探索中に詰みノードに達したとき
            return False
        for move in tsr.moves:
            self.pos.do_move(move)
        move_list = self.pos.generate_move_list()
        put_item = False
        if len(move_list) == 0:
            # 新しいノードが詰みだった時
            # DNNによる評価は不要のため、直ちにゲーム木に追加
            mate_node = TreeNode(self.tree_config, tsr.final_node, tsr.final_edge_index, move_list, -1.0, None)
            tsr.final_node.children[tsr.final_edge_index] = mate_node
        else:
            # 詰みでないので、評価を予約
            dnn_input = self.dnn_converter.get_board_array(self.pos)
            eval_wait_item = EvalWaitItem()
            eval_item = EvalItem()
            eval_item.dnn_input = dnn_input
            eval_wait_item.parent = tsr.final_node
            eval_wait_item.parent_edge_index = tsr.final_edge_index
            eval_wait_item.move_list = move_list
            eval_wait_item.move_indices = np.array(
                [self.dnn_converter.get_move_index(self.pos, move) for move in move_list])
            self.evaluator.put(eval_item, eval_wait_item)
            put_item = True
        for j in range(len(tsr.moves)):
            self.pos.undo_move()
        return put_item

    def _generate_root_node(self) -> TreeNode:
        """
        ルートノードを作成
        :return:
        """
        move_list = self.pos.generate_move_list()
        if len(move_list) == 0:
            # RESIGN
            return None
        dnn_input = self.dnn_converter.get_board_array(self.pos)
        eval_item = EvalItem()
        eval_item.dnn_input = dnn_input
        self.evaluator.put(eval_item, "root")
        self.evaluator.flush()
        move_indices = np.array([self.dnn_converter.get_move_index(self.pos, move) for move in move_list])
        while True:
            result_item, tag = self.evaluator.get(True)
            if tag == "root":
                # root nodeの評価結果
                return TreeNode(self.tree_config, None, 0, move_list,
                                result_item.score, result_item.move_probs[move_indices])
            else:
                logger.warning("Mismatch result for root node")

    def _make_strategy(self, usi_info_writer: UsiInfoWriter):
        """
        1手展開した結果に対し、評価関数を呼び出して手を決定する
        :return:
        """
        root_node = self._generate_root_node()
        logger.info("Generated root node")
        if root_node is None:
            if self.kifu_gen:
                return Move.MOVE_RESIGN, 0.0, None, None
            else:
                return Move.MOVE_RESIGN

        put_nodes = 0
        completed_nodes = 0
        dup_nodes = 0
        while completed_nodes < self.nodes:
            # TODO evaluatorのblock条件をみなおしてパフォーマンス改善
            if put_nodes < self.nodes and self.evaluator.pending_count() < 2:
                if not self._search_once(root_node):
                    # 評価不要ノードだったら直ちに完了とみなす
                    completed_nodes += 1
                put_nodes += 1
                if put_nodes == self.nodes:
                    self.evaluator.flush()
            try:
                result_item, eval_wait_item = self.evaluator.get(self.evaluator.pending_count() >= 2)  # type: Tuple[ResultItem, EvalWaitItem]

                completed_nodes += 1
                if eval_wait_item.parent.children[eval_wait_item.parent_edge_index] is None:
                    new_node = TreeNode(self.tree_config, eval_wait_item.parent, eval_wait_item.parent_edge_index,
                                        eval_wait_item.move_list, result_item.score,
                                        result_item.move_probs[eval_wait_item.move_indices])
                    eval_wait_item.parent.children[eval_wait_item.parent_edge_index] = new_node
                else:
                    eval_wait_item.parent._restore_virtual_loss(eval_wait_item.parent_edge_index)
                    dup_nodes += 1
                    # logger.warning("Duplicate new node; discard")
            except queue.Empty:
                pass
        logger.info(f"All nodes evaluation complete, nodes={completed_nodes}, dup={dup_nodes}")
        root_node.depth_stat()
        best_move, prob = root_node.play()
        usi_info_writer.write_string(f"{best_move.to_usi_string()}({int(prob*100)}%)")
        if self.kifu_gen:
            # bestmove, ルートの勝敗スコア, 指し手リスト、各指し手の選択回数
            return best_move, root_node.score, root_node.move_list, root_node.value_n
        else:
            return best_move

    @util.release_gpu_memory_pool
    def go(self, usi_info_writer: UsiInfoWriter, go_receive_time: float, btime: Optional[int] = None,
           wtime: Optional[int] = None, byoyomi: Optional[int] = None, binc: Optional[int] = None,
           winc: Optional[int] = None):
        move = self._make_strategy(usi_info_writer)
        return move.to_usi_string()

    def quit(self):
        self._close_evaluator()

    def gameover(self, result: str):
        self.evaluator.discard_pending_batches()

    def _close_evaluator(self):
        if self.evaluator is not None:
            self.evaluator.terminate()
            self.evaluator = None
