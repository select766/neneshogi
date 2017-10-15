"""
DNN評価値関数による、Monte Carlo Softmax Search Playerの実装
原型は「芝浦将棋Softmax」(WCSC27)のアルゴリズム。
GPUでバッチ処理するための変更を加えている。

DNNから出てくる方策自体は使わず、静的評価値のみ使用。

iteration単位で処理。
1回のiteration:
- ゲーム木をルートノードから末端までたどる。
  - 注目ノードの子ノードすべての評価値を用いて、softmaxによりランダムに子ノードを選択して進む。
- 末端ノードでは
  - 合法手すべてに対応する次の局面の子ノードを作成する。
  - すべての子ノードの静的評価値を計算する。(DNNでバッチ処理)
- 末端ノードからルートノードまで
  - 現在の子ノードの評価値と実現確率を用いて、注目ノードの評価値を更新する。

時間切れになったら、ルートノードから一番実現確率の高い子ノードを選択して手を指す。

qsearch(node, d)
- nodeに対する合法手を列挙。(同時に詰みかどうかの判定)
- 合法手のうち静止探索に関するものを抽出し、子ノードを作成。評価値計算予約。
- d-1>0のとき、各子ノードについて、
  - move
  - qsearch(child, d-1)
  - unmove
"""

import random
from typing import Dict, Optional, List

from logging import getLogger

import time

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

    def __init__(self, model: chainer.Chain, gpu: int, batchsize: int):
        self.items = []
        self.model = model
        self.gpu = gpu
        self.batchsize = batchsize
        self.resolve_count = 0
        self.softmax_temperature = 1.0

    def append(self, item: ValueProxy):
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
        # なぜか確保したメモリが再使用されずout of memoryになる問題へのworkaround
        chainer.cuda.memory_pool.free_all_blocks()


class GameTreeNode:
    is_leaf: bool
    is_mated: bool
    children: Dict[Move, "GameTreeNode"]
    children_order: List[Move]
    qsearch_children: Dict[Move, "GameTreeNode"]
    last_move: Move
    pv: Move
    value_proxy: ValueProxy
    value: float
    softmax_temperature: float

    def __init__(self, pos: Position, last_move: Move, value_proxy_batch: ValueProxyBatch, qsearch_remain_depth: int):
        self.children = {}
        self.children_order = []
        self.pv = None
        self.last_move = last_move
        self.is_leaf = True
        self.is_mated = False
        self.value_proxy = ValueProxy(pos)
        self.qsearch_children = {}
        value_proxy_batch.append(self.value_proxy)
        self.softmax_temperature = value_proxy_batch.softmax_temperature

        # 静止探索のツリーを作成
        if qsearch_remain_depth > 0:
            move_list = pos.generate_move_list()
            if pos.in_check():
                # 王手の時はすべての手
                move_list_filtered = move_list
            else:
                # 王手でないときは、last_moveと行先が同じ手
                move_list_filtered = [move for move in move_list if move.move_to == last_move.move_to]
            for move in move_list_filtered:
                undo_info = pos.do_move(move)
                child_node = GameTreeNode(pos, move, value_proxy_batch, qsearch_remain_depth - 1)
                self.qsearch_children[move] = child_node
                pos.undo_move(undo_info)

    def update_value(self):
        """
        葉ノードの場合は静的評価値をvalueに設定。
        内部ノードの場合は子ノードのsoftmax評価値をvalueに設定。
        :return:
        """
        if self.is_leaf:
            return self.update_qsearch_value()
        else:
            node_values = []
            max_move = None
            max_value = -100.0
            for move, node in self.children.items():
                node_value = -node.value
                if node_value > max_value:
                    max_value = node_value
                    max_move = move
                node_values.append(node_value)
            self.pv = max_move
            node_values = np.array(node_values, dtype=np.float32)
            exp_values = np.exp((node_values - np.max(node_values)) / self.softmax_temperature)
            self.value = np.sum(exp_values * node_values) / np.sum(exp_values)
        return self.value

    def update_qsearch_value(self) -> float:
        """
        自ノードの静的評価+静止探索の評価値をvalueに設定し返す
        :return:
        """
        if self.is_mated:
            self.value = -50.0
            return self.value
        assert self.value_proxy.resolved
        max_move = None
        max_value = self.value_proxy.value  # 何もしない状態の評価
        for move, node in self.qsearch_children.items():
            node_value = -node.update_qsearch_value()
            if node_value > max_value:
                max_value = node_value
                max_move = move
        self.pv = max_move
        self.value = max_value
        return max_value

    def sample_move(self, best=False) -> Move:
        """
        実現確率に従って次の手を選択する。
        :return:
        """
        assert not self.is_leaf
        node_values = []
        moves = []
        for move, node in self.children.items():
            node_value = -node.value
            node_values.append(node_value)
            moves.append(move)
        node_values = np.array(node_values, dtype=np.float32)
        exp_values = np.exp((node_values - np.max(node_values)) / self.softmax_temperature)
        move_probs = exp_values / np.sum(exp_values)
        if best:
            chosen_move_index = np.argmax(move_probs)
        else:
            chosen_move_index = np.random.choice(move_probs.size, p=move_probs)
        chosen_move = moves[chosen_move_index]
        return chosen_move

    def get_pv(self) -> List[Move]:
        """
        読み筋を出力。update_valueの後に呼び出し可能となる。
        :return:
        """
        if self.pv is None:
            return []
        else:
            if self.is_leaf:
                return [self.pv] + self.qsearch_children[self.pv].get_pv()
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

    def expand_child(self, pos: Position, value_proxy_batch: ValueProxyBatch, qsearch_depth: int):
        """
        現在葉ノードである場合に、子ノードをすべて作成する。(非静止探索)
        実現確率により子ノードをソートする。
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
            move = move_list[i]
            # 駒を取る手は必ず入れる
            if pos.board[move.move_to] != 0:
                move_prob = 1.0
            score_moves.append((-move_prob, i, move))  # iを入れることで、moveの比較が起こらないようにする
        score_moves.sort()  # move_probが大きい順に並び替え
        self.children_order = [item[2] for item in score_moves]

        for move in self.children_order:
            undo_info = pos.do_move(move)
            child_node = GameTreeNode(pos, move, value_proxy_batch, qsearch_depth)
            self.children[move] = child_node
            pos.undo_move(undo_info)
        self.is_leaf = False
        self.value_proxy = None  # もう静的評価値・方策は使わないので解放
        self.qsearch_children = None  # 葉ノードではなくなったので静止探索データを解放


class MonteCarloSoftmaxPlayer(Engine):
    pos: Position
    model: chainer.Chain
    depth: int
    batchsize: int
    gpu: int
    value_proxy_batch: ValueProxyBatch
    qsearch_depth: int
    nodes_count: int  # ある局面の探索開始からのノード数
    book: Book
    mate_searcher: MateSearcher
    softmax_temperature: float

    def __init__(self):
        self.pos = Position()
        self.model = None
        self.gpu = -1
        self.depth = 1
        self.batchsize = 256
        self.value_proxy_batch = None
        self.qsearch_depth = 0
        self.nodes_count = 0
        self.book = None
        self.mate_searcher = None
        self.softmax_temperature = 1.0

    @property
    def name(self):
        return "NeneShogi MonteCarloSoftmax"

    @property
    def author(self):
        return "select766"

    def get_options(self):
        return {"model_path": "filename default <empty>",
                "book": "filename default book/standard_book.db",
                "gpu": "spin default -1 min -1 max 0",
                "depth": "spin default 1 min 1 max 100",
                "qsearch_depth": "spin default 0 min 0 max 5",
                "softmax_temperature": "string default 1"}

    def isready(self, options: Dict[str, str]):
        self.gpu = int(options["gpu"])
        self.depth = int(options["depth"])
        self.qsearch_depth = int(options["qsearch_depth"])
        self.softmax_temperature = float(options["softmax_temperature"])
        book_path = options["book"]
        if len(book_path) > 0:
            self.book = Book()
            self.book.load(util.strip_path(book_path))
        self.model = load_model(options["model_path"])
        if self.gpu >= 0:
            chainer.cuda.get_device_from_id(self.gpu).use()
            self.model.to_gpu()
        self.value_proxy_batch = ValueProxyBatch(self.model, self.gpu, self.batchsize)
        self.value_proxy_batch.softmax_temperature = self.softmax_temperature
        self.mate_searcher = MateSearcher()
        self.mate_searcher.run()
        # TODO: ここで一度NNを走らせて、CUDAカーネルの初期化をさせたほうがよい

    def position(self, command: str):
        self.pos.set_usi_position(command)

    def do_search_recursion(self, node: GameTreeNode, remain_depth: int, alpha: float, beta: float):
        if remain_depth <= 1:
            return self.do_search_eval_children(node)
        node.pv = None
        node.expand_child(self.pos, self.value_proxy_batch, 0)
        self.value_proxy_batch.resolve()
        if node.is_mated:
            return -50.0

        for move in node.children_order:
            child = node.children[move]
            undo_info = self.pos.do_move(move)
            child_val = -self.do_search_recursion(child, remain_depth - 1, -beta, -alpha)
            if child_val > alpha:
                alpha = child_val
                node.pv = move
            self.pos.undo_move(undo_info)
            if alpha > beta:
                return alpha
        return alpha

    def do_search_eval_children(self, node: GameTreeNode):
        node.expand_child(self.pos, self.value_proxy_batch, self.qsearch_depth)
        self.value_proxy_batch.resolve()
        return node.get_value()

    def do_search_root(self, usi_info_writer: UsiInfoWriter, tree_root: GameTreeNode, depth: int):
        """
        1 iteration木を更新する
        :return:
        """
        logger.info(f"generating game tree of depth {depth}")
        self.value_proxy_batch.resolve_count = 0
        # 末端ノードまで進む
        cur_node = tree_root
        node_stack = [tree_root]
        undo_stack = []
        while not cur_node.is_leaf:
            move = cur_node.sample_move()
            undo_stack.append(self.pos.do_move(move))
            cur_node = cur_node.children[move]
            node_stack.append(cur_node)
        # 木を掘り下げる
        cur_node.expand_child(self.pos, self.value_proxy_batch, self.qsearch_depth)
        self.value_proxy_batch.resolve()
        for child in cur_node.children.values():
            child.update_value()
        while len(node_stack) > 0:
            cur_node = node_stack.pop()
            cur_node.update_value()
            if len(undo_stack) > 0:
                self.pos.undo_move(undo_stack.pop())
        self.nodes_count += self.value_proxy_batch.resolve_count
        # 読み筋を計算
        pv = tree_root.get_pv()
        logger.info("done")
        if len(pv) == 0:
            return "resign"

        usi_info_writer.write_pv(pv=pv, depth=int(depth), nodes=self.nodes_count, score_cp=int(tree_root.value * 600))
        return pv[0].to_usi_string()

    def generate_tree_root(self) -> GameTreeNode:
        """
        ルートノードを作成し、静的評価値を計算する（今のところ静的評価値は不要）
        :return:
        """
        tree_root = GameTreeNode(self.pos, None, self.value_proxy_batch, 0)
        self.value_proxy_batch.resolve()
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
                search_time += (mytime - myinc) / 1000.0 / 50 + myinc / 1000.0
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
            book_move = self.book.get_move(self.pos)
            if book_move is not None:
                book_move.write_pv(usi_info_writer)
                return book_move.move.to_usi_string()
        move_str = "resign"
        self.mate_searcher.stop_signal.value = 0
        self.mate_searcher.command_queue.put(MateSearcherCommand.go(self.pos))
        tree_root = self.generate_tree_root()
        for cur_depth in range(1, self.depth + 1):
            move_str = self.do_search_root(usi_info_writer, tree_root, cur_depth)
            if time.time() >= self.search_end_time:
                logger.info("search timeup")
                break
        self.mate_searcher.stop_signal.value = 1
        mate_result = self.mate_searcher.response_queue.get()
        logger.info(f"mate result: {mate_result.params}")
        return move_str

    def gameover(self, result: str):
        self.mate_searcher.quit()
        self.mate_searcher = None
