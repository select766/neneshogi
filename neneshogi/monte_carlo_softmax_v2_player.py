"""
DNN評価値関数による、Monte Carlo Softmax Search Playerの実装
原型は「芝浦将棋Softmax」(WCSC27)のアルゴリズム。
GPUでバッチ処理するための変更を加えている。

DNNから出てくる方策自体は使わず、静的評価値のみ使用。

局面と評価値のテーブルを用いて、探索結果を次の手の思考でも利用可能とする。

3種類の処理が存在。
木の探索
- ゲーム木をルートノードから末端までたどる。
  - 注目ノードの子ノードすべての評価値を用いて、softmaxによりランダムに子ノードを選択して進む。
- 末端ノード（少なくとも1つの子ノードが存在しないノード）では
  - 合法手に対応する子ノードすべてについて、静止探索ツリーを生成しNNキューに入れる。
  - 詰み局面なら局面と評価値を評価値キューに入れる。

静的評価値計算
- 静止探索ツリーをNNキューから取り出す。
- GPUで局面の評価値を計算。
- alpha-betaでルート局面の評価値を計算。
- 評価値キューに入れる。

評価値の再帰的更新
- 評価値キューからアイテムを取り出す。
- undoリストに沿って評価値テーブルの値を更新する。

時間切れになったら、ルートノードから一番実現確率の高い子ノードを選択して手を指す。

qsearch(node, d)
- nodeに対する合法手を列挙。(同時に詰みかどうかの判定)
- 合法手のうち静止探索に関するものを抽出し、子ノードを作成。
- d-1>0のとき、各子ノードについて、
  - move
  - qsearch(child, d-1)
  - unmove
"""

import random
from typing import Dict, Optional, List

from logging import getLogger
import multiprocessing

import time

from neneshogi.move import UndoMoveInfo

logger = getLogger(__name__)

import numpy as np
import chainer

from .position import Position, Color, Square, Piece, Move
from .engine import Engine
from .usi_info_writer import UsiInfoWriter
from .train_config import load_model
from .book import BookMove, Book
from .mate_searcher import MateSearcher, MateSearcherCommand, MateSearcherResponse
from . import util


class NNInfo:
    gpu: int
    model_path: str
    batch_size: int

    def __init__(self, model_path: str, gpu: int, batch_size: int):
        self.model_path = model_path
        self.gpu = gpu
        self.batch_size = batch_size


class NNSearchProcess:
    nn_info: NNInfo
    nn_queue: multiprocessing.Queue
    value_queue: multiprocessing.Queue

    def __init__(self, nn_info: NNInfo,
                 nn_queue: multiprocessing.Queue,
                 value_queue: multiprocessing.Queue):
        self.nn_info = nn_info
        self.nn_queue = nn_queue
        self.value_queue = value_queue
        self.model = load_model(nn_info.model_path)
        if self.nn_info.gpu >= 0:
            chainer.cuda.get_device_from_id(self.nn_info.gpu).use()
            self.model.to_gpu()

    def run(self):
        while True:
            q_tree = self.nn_queue.get()
            if q_tree is None:
                # signal of exit
                break
            raise NotImplementedError


def run_nn_search_process(nn_info: NNInfo,
                          nn_queue: multiprocessing.Queue,
                          value_queue: multiprocessing.Queue
                          ):
    nn_search = NNSearchProcess(nn_info, nn_queue, value_queue)
    nn_search.run()


class NNEvalItem:
    def __init__(self, q_tree: QTreeNode, undo_stack: List[UndoMoveInfo]):
        self.q_tree = q_tree
        self.undo_stack = undo_stack


class NNValueItem:
    def __init__(self, pos_key: PositionKey, static_value: float, undo_stack: List[UndoMoveInfo]):
        self.pos_key = pos_key
        self.static_value = static_value
        self.undo_stack = undo_stack


class PositionKey:
    """
    置換表のキーとして使うPosition
    """
    _pos: Position

    def __init__(self, pos: Position):
        self._pos = PositionKey._copy_pos(pos)

    def get_pos(self):
        return PositionKey._copy_pos(self._pos)

    @staticmethod
    def _copy_pos(pos: Position) -> Position:
        dst = Position()
        dst.board[:] = pos.board
        dst.hand[:] = pos.hand
        dst.side_to_move = pos.side_to_move
        dst.game_ply = pos.game_ply
        return dst

    def __eq__(self, other):
        return self._pos.eq_board(other)

    def __hash__(self):
        return hash(self._pos.board.tobytes()) ^ hash(self._pos.hand.tobytes()) ^ self._pos.side_to_move


class TTValue:
    """
    置換表の値
    """
    static_value: float
    propagate_value: float

    def __init__(self, static_value: float, propagate_value: Optional[float]):
        self.static_value = static_value
        self.propagate_value = propagate_value

    @property
    def value(self) -> float:
        if self.propagate_value is None:
            return self.static_value
        else:
            return self.propagate_value


class QTreeNode:
    is_mated: bool
    pos_key: PositionKey
    static_value: float
    qsearch_children: Dict[Move, "QTreeNode"]
    last_move: Move
    pv: Move

    def __init__(self, pos: Position, last_move: Move, qsearch_remain_depth: int):
        self.pv = None
        self.last_move = last_move
        self.is_mated = False
        self.qsearch_children = {}
        self.pos_key = PositionKey(pos)
        self.static_value = None

        # 静止探索のツリーを作成
        if qsearch_remain_depth > 0:
            move_list = pos.generate_move_list()
            if len(move_list) == 0:
                self.is_mated = True
            else:
                if pos.in_check():
                    # 王手の時はすべての手
                    move_list_filtered = move_list
                else:
                    # 王手でないときは、last_moveと行先が同じ手
                    move_list_filtered = [move for move in move_list if move.move_to == last_move.move_to]
                for move in move_list_filtered:
                    undo_info = pos.do_move(move)
                    child_node = QTreeNode(pos, move, qsearch_remain_depth - 1)
                    self.qsearch_children[move] = child_node
                    pos.undo_move(undo_info)

    def enum_items(self):
        """
        木構造の全ノードを列挙する。
        :return:
        """
        yield self
        for child in self.qsearch_children.values():
            for item in child.enum_items():
                yield item

    def get_value(self):
        """
        minimax法により評価値および読み筋を計算する。
        :return:
        """
        if self.is_mated:
            self.static_value = -50.0
            return self.static_value
        max_move = None
        assert self.static_value is not None
        max_value = self.static_value  # 何もしない状態の評価
        for move, node in self.qsearch_children.items():
            node_value = -node.get_value()
            if node_value > max_value:
                max_value = node_value
                max_move = move
        self.pv = max_move
        return max_value


class MonteCarloSoftmaxV2Player(Engine):
    pos: Position
    max_nodes: int
    qsearch_depth: int
    nodes_count: int  # ある局面の探索開始からのノード数
    ttable: Dict[PositionKey, TTValue]  # 置換表
    book: Book
    # mate_searcher: MateSearcher
    softmax_temperature: float
    nn_search_process: multiprocessing.Process
    nn_queue: multiprocessing.Queue
    value_queue: multiprocessing.Queue

    def __init__(self):
        self.pos = Position()
        self.model = None
        self.max_nodes = 1000
        self.value_proxy_batch = None
        self.qsearch_depth = 0
        self.nodes_count = 0
        self.book = None
        self.mate_searcher = None
        self.softmax_temperature = 1.0

    @property
    def name(self):
        return "NeneShogi MonteCarloSoftmaxV2"

    @property
    def author(self):
        return "select766"

    def get_options(self):
        return {"model_path": "filename default <empty>",
                "book": "filename default book/standard_book.db",
                "gpu": "spin default -1 min -1 max 0",
                "max_nodes": "spin default 1000 min 1 max 10000000",
                "qsearch_depth": "spin default 0 min 0 max 5",
                "softmax_temperature": "string default 1"}

    def isready(self, options: Dict[str, str]):
        self.max_nodes = int(options["max_nodes"])
        self.qsearch_depth = int(options["qsearch_depth"])
        self.softmax_temperature = float(options["softmax_temperature"])
        book_path = options["book"]
        if len(book_path) > 0:
            self.book = Book()
            self.book.load(util.strip_path(book_path))
        # NN処理プロセスの起動
        self.nn_queue = multiprocessing.Queue(256)
        self.value_queue = multiprocessing.Queue()
        nn_info = NNInfo(gpu=int(options["gpu"]), model_path=options["model_path"], batch_size=256)
        self.nn_search_process = multiprocessing.Process(target=run_nn_search_process,
                                                         args=(nn_info, self.nn_queue, self.value_queue))

    def position(self, command: str):
        self.pos.set_usi_position(command)

    def sample_move_index(self, child_values, best=False) -> int:
        node_values = -child_values
        exp_values = np.exp((node_values - np.max(node_values)) / self.softmax_temperature)
        move_probs = exp_values / np.sum(exp_values)
        if best:
            chosen_move_index = np.argmax(move_probs)
        else:
            chosen_move_index = np.random.choice(move_probs.size, p=move_probs)
        return chosen_move_index

    def search_reserve(self, usi_info_writer: UsiInfoWriter):
        """
        末端まで探索して評価値計算の予約を行う
        self.posは探索開始局面をさしているとする
        :return:
        """
        # 末端ノードまで進む
        # 末端では、子ノードの評価値がない
        undo_stack = []
        is_pos_terminal = False
        last_move = None  # 末端局面に至る直前の手(recaptureで利用)
        while not is_pos_terminal:
            moves = self.pos.generate_move_list()
            if len(moves) == 0:
                # 詰み局面
                self.ttable[PositionKey(self.pos)] = TTValue(-50.0, None)
                if len(undo_stack) > 0:
                    self.pos.undo_move(undo_stack.pop())
                return
            child_values = np.zeros((len(moves),), dtype=np.float32)
            for i, move in enumerate(moves):
                undo_info = self.pos.do_move(move)
                pk = PositionKey(self.pos)
                if pk not in self.ttable:
                    # 子ノードが置換表にないので、self.posの1手前は末端ノード
                    self.pos.undo_move(undo_info)
                    is_pos_terminal = True
                    break
                else:
                    v = self.ttable[pk]
                    child_values[i] = v.value
                    self.pos.undo_move(undo_info)
            else:
                # すべての子ノードがあったので、確率的に次のノードを選択
                move_index = self.sample_move_index(child_values)
                move = moves[move_index]
                last_move = move
                undo_stack.append(self.pos.do_move(move))

        # 静止探索ツリーを作成
        q_tree = QTreeNode(self.pos, last_move, self.qsearch_depth)
        # 評価値計算を予約
        self.nn_queue.put(NNEvalItem(q_tree, undo_stack))
        if len(undo_stack) > 0:
            self.pos.undo_move(undo_stack.pop())

        return

    def update_tree_values(self, nn_value: NNValueItem):
        """
        新規ノードへの評価値を受け取り、ルートノードからの経路上の評価値を更新する。
        :return:
        """

    def find_pv(self):
        """
        現在の評価に従って読み筋を生成する。
        :return:
        """

    def generate_tree_root(self) -> PositionKey:
        """
        ルートノードを作成する
        :return:
        """
        tree_root = PositionKey(self.pos)
        # 前回の探索によりすでにノードが存在する場合あり
        if tree_root not in self.ttable:
            self.ttable[tree_root] = TTValue(0.0, None)
        return tree_root

    def calculate_search_time(self, btime: Optional[int], wtime: Optional[int],
                              byoyomi: Optional[int], binc: Optional[int], winc: Optional[int]) -> float:
        """
        今回の思考時間を決定する。
        :param btime:
        :param wtime:
        :param byoyomi:
        :param binc:
        :param winc:
        :return:
        """
        search_time = 0.0
        if self.pos.side_to_move == Color.BLACK:
            mytime = btime
            myinc = binc
        else:
            mytime = wtime
            myinc = winc

        if mytime is not None:
            if myinc is not None:
                search_time += (mytime - myinc) / 1000.0 / 25 + myinc / 1000.0
            else:
                search_time += mytime / 1000.0 / 50
        if byoyomi is not None:
            search_time += byoyomi / 1000.0

        margin = 2.0
        min_time = 2.0
        search_time = max(search_time - margin, min_time)
        logger.info(f"Scheduled search time: {search_time}sec")
        return search_time

    @util.release_gpu_memory_pool
    def go(self, usi_info_writer: UsiInfoWriter, btime: Optional[int] = None, wtime: Optional[int] = None,
           byoyomi: Optional[int] = None, binc: Optional[int] = None, winc: Optional[int] = None):
        self.search_end_time = time.time() + self.calculate_search_time(btime, wtime, byoyomi, binc, winc)

        self.nodes_count = 0

        if self.book is not None:
            book_move = self.book.get_move(self.pos)  # type: BookMove
            if book_move is not None:
                book_move.write_pv(usi_info_writer)
                return book_move.move.to_usi_string()
        move_str = "resign"
        # self.mate_searcher.stop_signal.value = 0
        # self.mate_searcher.command_queue.put(MateSearcherCommand.go(self.pos))
        tree_root = self.generate_tree_root()
        for cur_depth in range(1, self.depth + 1):
            move_str = self.do_search_root(usi_info_writer, tree_root, cur_depth)
            if time.time() >= self.search_end_time:
                logger.info("search timeup")
                break
        # self.mate_searcher.stop_signal.value = 1
        # mate_result = self.mate_searcher.response_queue.get()
        # logger.info(f"mate result: {mate_result.params}")
        return move_str

    def gameover(self, result: str):
        logger.info("joining nn process")
        self.nn_queue.put(None)
        self.nn_search_process.join()
        logger.info("joined nn process")
        # self.mate_searcher.quit()
        # self.mate_searcher = None
