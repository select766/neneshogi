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
import queue
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
    next_serial: int
    forward_complete_len: int
    serial_dnn_input: Dict[int, np.ndarray]
    serial_qtree_node: Dict[int, "QTreeNode"]
    pending_eval_items: List["NNEvalItem"]
    force_flush: bool

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
        self.next_serial = 0
        self.forward_complete_len = 0
        self.serial_dnn_input = {}
        self.serial_qtree_node = {}
        self.pending_eval_items = []
        self.force_flush = False

    def run(self):
        while True:
            eval_item = self.nn_queue.get()  # type: NNEvalItem
            if eval_item is None:
                # signal of exit
                logger.info("Received exit message")
                self.value_queue.put(None)
                break
            # TODO 静止探索
            self._add_to_pending(eval_item)
            logger.info(f"next_serial: {self.next_serial}, forward_complete_len: {self.forward_complete_len}, flush: {self.force_flush}")
            while (self.next_serial - self.forward_complete_len >= self.nn_info.batch_size) or \
                    (self.force_flush and self.next_serial > self.forward_complete_len):
                # バッチサイズ分データが集まったので計算
                self._process_batch()
            self.force_flush = False

    def _add_to_pending(self, eval_item: "NNEvalItem"):
        # 評価する行列単位に分割して処理待ちバッファに入れる
        for q_tree in eval_item.q_trees:
            self.serial_dnn_input[self.next_serial] = q_tree.dnn_board
            self.serial_qtree_node[self.next_serial] = q_tree
            self.next_serial += 1
        eval_item.final_serial = self.next_serial - 1  # このシリアル番号まで処理が完了したら、このリクエストの処理が完了
        self.pending_eval_items.append(eval_item)
        if eval_item.flush:
            self.force_flush = True

    def _process_batch(self):
        dnn_inputs = []
        for serial in range(self.forward_complete_len, min(self.forward_complete_len + self.nn_info.batch_size, self.next_serial)):
            dnn_inputs.append(self.serial_dnn_input.pop(serial))
        cur_batchsize = len(dnn_inputs)
        static_values = self._evaluate(dnn_inputs)
        for serial in range(self.forward_complete_len, self.forward_complete_len + cur_batchsize):
            self.serial_qtree_node.pop(serial).static_value = static_values.pop(0)
        self.forward_complete_len += cur_batchsize
        while len(self.pending_eval_items) > 0:
            if self.pending_eval_items[0].final_serial < self.forward_complete_len:
                # 対応する全Qtreeのstatic_valueが計算済み
                complete_eval_item = self.pending_eval_items.pop(0)
                item_values = [q_tree.static_value for q_tree in complete_eval_item.q_trees]
                self.value_queue.put(NNValueItem(item_values, complete_eval_item.side_id))
            else:
                break


    def _evaluate(self, dnn_inputs: List[np.ndarray]) -> List[float]:
        with chainer.using_config('train', False):
            with chainer.using_config('enable_backprop', False):
                dnn_input = np.concatenate(dnn_inputs, axis=0)
                if self.nn_info.gpu >= 0:
                    dnn_input = chainer.cuda.to_gpu(dnn_input)
                model_output_var_move, model_output_var_value = self.model.forward(dnn_input)
                model_output_value = chainer.cuda.to_cpu(model_output_var_value.data)
        return model_output_value.tolist()  # type: List[float]


def run_nn_search_process(nn_info: NNInfo,
                          nn_queue: multiprocessing.Queue,
                          value_queue: multiprocessing.Queue
                          ):
    try:
        nn_search = NNSearchProcess(nn_info, nn_queue, value_queue)
        import os
        from . import config
        if os.environ.get("NENESHOGI_PROFILE", "0") == "1":
            import cProfile
            import time

            profile_path = os.path.join(config.PROFILE_DIR, f"cprofile_{time.strftime('%Y%m%d%H%M%S')}.nn_process.bin")
            cProfile.runctx('nn_search.run()', globals(), locals(), filename=profile_path)
        else:
            nn_search.run()
    except Exception as ex:
        logger.exception("Unhandled error")


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
    """
    静止探索を含めた、1局面を評価するのに必要な情報。
    """
    is_mated: bool
    dnn_board: np.ndarray
    static_value: float
    qsearch_children: Dict[Move, "QTreeNode"]
    last_move: Move
    pv: Move

    def __init__(self, pos: Position, last_move: Move, qsearch_remain_depth: int):
        self.pv = None
        self.last_move = last_move
        self.is_mated = False
        self.qsearch_children = {}
        #self.dnn_board = self._make_dnn_input(pos)
        self.dnn_board = np.empty((1, 61, 9, 9), dtype=np.float32)
        pos.make_dnn_input(0, self.dnn_board)
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
        board = pos_from_side.board
        hand = pos_from_side.hand
        # 盤上の駒
        for sq in range(Square.SQ_NB):
            piece = board[sq]
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
                hand_count = hand[color][i]
                ch = color * 7 + 28 + i
                ary[ch, :] = hand_count
        # 段・筋
        for sq in range(Square.SQ_NB):
            ary[Square.rank_of(sq) + 42, sq] = 1.0
            ary[Square.file_of(sq) + 51, sq] = 1.0
        # 定数1
        ary[60, :] = 1.0
        return ary.reshape((1, 61, 9, 9))


class NNEvalItem:
    def __init__(self, q_trees: List[QTreeNode], side_id: int, flush: bool):
        self.q_trees = q_trees
        self.side_id = side_id
        self.flush = flush


class NNValueItem:
    def __init__(self, static_values: List[float], side_id: int):
        self.static_values = static_values
        self.side_id = side_id


class NNSideItem:
    """
    NN処理の結果を利用するための付属情報。対応するPositionなど。
    """

    q_tree_pos_list: List[int]
    parent_pos: Position
    undo_stack: List[UndoMoveInfo]

    def __init__(self, q_tree_pos_list: List[int], parent_pos: Position, undo_stack: List[UndoMoveInfo]):
        self.q_tree_pos_list = q_tree_pos_list
        self.parent_pos = parent_pos
        self.undo_stack = undo_stack


class MonteCarloSoftmaxV2Player(Engine):
    pos: Position
    max_nodes: int
    qsearch_depth: int
    nodes_count: int  # ある局面の探索開始からのノード数
    ttable: Dict[int, TTValue]  # 置換表
    book: Book
    gpu: int
    # mate_searcher: MateSearcher
    softmax_temperature: float
    nn_search_process: multiprocessing.Process
    nn_queue: multiprocessing.Queue
    value_queue: multiprocessing.Queue
    side_buffer: Dict[int, NNSideItem]
    root_side_id: int  # root局面を評価予約した際のid。結果が返ったらNoneにする。

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
                "batch_size": "spin default 256 min 1 max 32768",
                "queue_size": "spin default 16 min 1 max 1024",
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
        self.nn_queue = multiprocessing.Queue(int(options["queue_size"]))
        self.value_queue = multiprocessing.Queue()
        self.gpu = int(options["gpu"])
        nn_info = NNInfo(gpu=self.gpu, model_path=options["model_path"], batch_size=int(options["batch_size"]))
        self.nn_search_process = multiprocessing.Process(target=run_nn_search_process,
                                                         args=(nn_info, self.nn_queue, self.value_queue))
        self.nn_search_process.start()
        self.ttable = {}
        self.side_buffer = {}
        self.root_side_id = None

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

    def search_reserve(self):
        """
        末端まで探索して評価値計算の予約を行う
        self.posは探索開始局面をさしているとする
        :return:
        """
        # 末端ノードまで進む
        # 末端では、子ノードの評価値がない
        undo_stack = []
        is_pos_terminal = False
        new_move_pos_list = []  # 末端ノードから1手先の手と局面リスト。すでに置換表にあるものは除く。
        depth = 0
        while depth < 100:  # 千日手ループ回避
            moves = self.pos.generate_move_list()
            if len(moves) == 0:
                # 詰み局面
                pk = self.pos.hash()
                if len(undo_stack) > 0:
                    # ルート局面自体が詰みだと親局面が設定できない
                    self.pos.undo_move(undo_stack.pop())
                    side_item = NNSideItem([pk], self.pos.copy(), undo_stack)
                    self.side_buffer[id(side_item)] = side_item
                    self.value_queue.put(NNValueItem([-50.0], id(side_item)))
                for undo_info in reversed(undo_stack):
                    self.pos.undo_move(undo_info)
                return
            child_values = np.zeros((len(moves),), dtype=np.float32)
            for i, move in enumerate(moves):
                undo_info = self.pos.do_move(move)
                pk = self.pos.hash()
                if pk not in self.ttable:
                    # 子ノードが置換表にないので、self.posの1手前は末端ノード
                    new_move_pos_list.append((move, self.pos.copy()))
                    self.pos.undo_move(undo_info)
                    is_pos_terminal = True
                else:
                    v = self.ttable[pk]
                    child_values[i] = v.value
                    self.pos.undo_move(undo_info)

            if is_pos_terminal:
                break
            # すべての子ノードがあったので、確率的に次のノードを選択
            move_index = self.sample_move_index(child_values)
            move = moves[move_index]
            undo_stack.append(self.pos.do_move(move))
            depth += 1

        # まだない子ノードそれぞれについて、静止探索ツリーを作成
        q_trees = []
        q_tree_pos_keys = []
        for last_move, new_pos in new_move_pos_list:
            q_tree = QTreeNode(new_pos, last_move, self.qsearch_depth)
            q_trees.append(q_tree)
            q_tree_pos_keys.append(new_pos.hash())
        # 評価値計算を予約
        side_item = NNSideItem(q_tree_pos_keys, self.pos.copy(), undo_stack)
        self.side_buffer[id(side_item)] = side_item
        is_root = False
        if len(undo_stack) == 0:
            # ルート局面を予約した
            self.root_side_id = id(side_item)
            is_root = True
        self.nn_queue.put(NNEvalItem(q_trees, id(side_item), flush=is_root))
        for undo_info in reversed(undo_stack):
            self.pos.undo_move(undo_info)

    def update_tree_values(self, nn_value: NNValueItem):
        """
        新規ノードへの評価値を受け取り、ルートノードからの経路上の評価値を更新する。
        :return:
        """
        if nn_value.side_id == self.root_side_id:
            # ルート局面の結果
            self.root_side_id = None
        side_item = self.side_buffer.pop(nn_value.side_id)  # type: NNSideItem

        # 静的評価値の登録
        for pk, sv in zip(side_item.q_tree_pos_list, nn_value.static_values):
            self.ttable[pk] = TTValue(sv, None)
            self.nodes_count += 1
        # 親ノードの更新
        undo_stack = side_item.undo_stack
        pos = side_item.parent_pos
        while True:
            # posの子ノードすべての評価値から、posの評価値を計算
            moves = pos.generate_move_list()
            child_values = []
            for move in moves:
                undo_info = pos.do_move(move)
                pk = pos.hash()
                value = -self.ttable[pk].value  # 自分の手番での評価値に変換
                pos.undo_move(undo_info)
                child_values.append(value)
            node_values = np.array(child_values, dtype=np.float32)
            exp_values = np.exp((node_values - np.max(node_values)) / self.softmax_temperature)
            prop_value = np.sum(exp_values * node_values) / np.sum(exp_values)
            self.ttable[pos.hash()].propagate_value = float(prop_value)
            if len(undo_stack) == 0:
                break
            pos.undo_move(undo_stack.pop())

    def find_pv(self):
        """
        現在の評価に従って読み筋を生成する。
        :return:
        """
        # 末端ノードまで進む
        # 末端では、子ノードの評価値がない
        undo_stack = []
        is_pos_terminal = False
        pv = []
        while len(pv) < 100:  # 千日手ループ回避
            moves = self.pos.generate_move_list()
            if len(moves) == 0:
                # 詰み局面
                break
            child_values = np.zeros((len(moves),), dtype=np.float32)
            for i, move in enumerate(moves):
                undo_info = self.pos.do_move(move)
                pk = self.pos.hash()
                if pk not in self.ttable:
                    # 子ノードが置換表にないので、self.posの1手前は末端ノード
                    self.pos.undo_move(undo_info)
                    is_pos_terminal = True
                else:
                    v = self.ttable[pk]
                    child_values[i] = v.value
                    self.pos.undo_move(undo_info)

            if is_pos_terminal:
                break
            # すべての子ノードがあったので、貪欲に次のノードを選択
            move_index = self.sample_move_index(child_values, best=True)
            move = moves[move_index]
            undo_stack.append(self.pos.do_move(move))
            pv.append(move)

        while len(undo_stack) > 0:
            self.pos.undo_move(undo_stack.pop())
        return pv

    def generate_tree_root(self) -> int:
        """
        ルートノードを作成する
        :return:
        """
        tree_root = self.pos.hash()
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
        self.search_start_time = time.time()
        self.search_end_time = self.search_start_time + self.calculate_search_time(btime, wtime, byoyomi, binc, winc)

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
        pv = []
        next_pv_update_time = self.search_start_time  # 次にPV計算をする時刻
        while True:
            self.search_reserve()
            while True:
                try:
                    # ルート局面の評価が来るまでは、次の探索をしても仕方ないので待つ
                    nn_value = self.value_queue.get(block=self.root_side_id is not None)
                    self.update_tree_values(nn_value)
                except queue.Empty:
                    break
            cur_time = time.time()
            timeup = cur_time >= self.search_end_time
            if cur_time > next_pv_update_time or timeup:
                pv = self.find_pv()
                if len(pv) > 0:
                    root_value = self.ttable[tree_root].value
                    usi_info_writer.write_pv(pv=pv,
                                             depth=len(pv),
                                             score_cp=int(root_value * 600),
                                             nodes=self.nodes_count,
                                             time=int((cur_time - self.search_start_time) * 1000))
                next_pv_update_time += 2.0
                if timeup:
                    logger.info("search timeup")
                    break
        # self.mate_searcher.stop_signal.value = 1
        # mate_result = self.mate_searcher.response_queue.get()
        # logger.info(f"mate result: {mate_result.params}")
        if len(pv) > 0:
            move_str = pv[0].to_usi_string()
        return move_str

    def gameover(self, result: str):
        self._join_nn_process()
        # self.mate_searcher.quit()
        # self.mate_searcher = None
        self.ttable = None
        self.side_buffer = None

    def quit(self):
        self._join_nn_process()

    def _join_nn_process(self):
        if self.nn_search_process is not None:
            logger.info("joining nn process")
            self.nn_queue.put(None)
            while self.value_queue.get() is not None:
                pass
            self.nn_search_process.join()
            logger.info("joined nn process")
            self.nn_search_process = None
