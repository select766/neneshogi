from collections import deque
from typing import Dict, Optional, List, Tuple
import queue
import multiprocessing
from abc import ABCMeta, abstractmethod

from logging import getLogger

import time

logger = getLogger(__name__)

import numpy as np
import chainer
import chainer.functions as F

from .position import Position, Color, Square, Piece, Move, PositionHelper
from .train_config import load_model


class EvaluatorConfig:
    model_path: str
    gpu: int


class EvalItem:
    eval_id: int
    dnn_input: np.ndarray


class ResultItem:
    eval_id: int
    move_probs: np.ndarray
    score: float


class EvaluatorBase(metaclass=ABCMeta):
    @abstractmethod
    def put(self, eval_item: EvalItem, tag: object) -> None:
        """
        プロセス内キューにEvalItemを追加する。
        満杯になればそれを1バッチとしてflush()を呼び出す。
        :param eval_item:
        :return:
        """
        pass

    @abstractmethod
    def flush(self) -> None:
        """
        GPUプロセスにEvalItemを送信する。
        :return:
        """
        pass

    @abstractmethod
    def pending_count(self) -> int:
        """
        GPUプロセス側で未処理のバッチ数を返す。
        :return:
        """
        pass

    def discard_pending_batches(self):
        """
        今までにputした、古い設定で得られた結果を捨てる
        :return:
        """
        pass

    @abstractmethod
    def get(self, block: bool) -> Tuple[ResultItem, object]:
        """
        GPUの処理結果を取得する。
        :param block:
        :return:
        """
        pass

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def terminate(self):
        pass


class EvaluatorSingleGPU(EvaluatorBase):
    """
    このインスタンスに対して1つのプロセスを生成し、1GPUで処理するEvaluator
    """
    eval_init_queue: multiprocessing.Queue
    eval_init_complete_event: multiprocessing.Event
    eval_queue: multiprocessing.Queue
    result_queue: multiprocessing.Queue
    eval_local_queue: List[EvalItem]
    received_items: deque
    tags: Dict[int, object]
    evaluator_process: multiprocessing.Process
    n_batch_put: int
    n_batch_get: int
    batch_size: int
    first_config: bool

    def __init__(self):
        self.eval_init_queue = multiprocessing.Queue()
        self.eval_init_complete_event = multiprocessing.Event()
        self.eval_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.eval_local_queue = []
        self.tags = {}
        self.evaluator_process = None
        self.n_batch_put = 0
        self.n_batch_get = 0
        self.batch_size = None
        self.first_config = True
        self.received_items = deque()

    def start(self):
        self.evaluator_process = multiprocessing.Process(target=evaluator_main,
                                                         args=(self.eval_init_queue, self.eval_init_complete_event,
                                                               self.eval_queue, self.result_queue),
                                                         daemon=True)
        self.evaluator_process.start()

        # この時点でevaluator processがちゃんと動き出すまで待つ(sys.stdinを読む前に)
        self.eval_init_complete_event.wait()
        self.eval_init_complete_event.clear()

    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size

    def set_config(self, evaluator_config: EvaluatorConfig):
        if self.first_config:
            self.first_config = False
        else:
            # いったん評価ループを終了させる
            self.eval_queue.put(None)
        self.discard_pending_batches()
        self.eval_init_queue.put(evaluator_config)
        self.eval_init_complete_event.wait()
        self.eval_init_complete_event.clear()

    def discard_pending_batches(self):
        """
        今までにputした、古い設定で得られた結果を捨てる
        :return:
        """
        self.flush()
        while self.n_batch_get < self.n_batch_put:
            self.result_queue.get()
            self.n_batch_get += 1

    def put(self, eval_item: EvalItem, tag: object):
        self.tags[id(tag)] = tag
        eval_item.eval_id = id(tag)
        self.eval_local_queue.append(eval_item)
        if len(self.eval_local_queue) >= self.batch_size:
            self.flush()

    def flush(self):
        if len(self.eval_local_queue) > 0:
            self.eval_queue.put(self.eval_local_queue)
            self.eval_local_queue = []
            self.n_batch_put += 1

    def pending_count(self):
        return self.n_batch_put - self.n_batch_get

    def get(self, block: bool):
        if len(self.received_items) == 0:
            self.received_items.extend(self.result_queue.get(block=block))
            self.n_batch_get += 1
        r_item = self.received_items.popleft()  # type: ResultItem
        tag = self.tags.pop(r_item.eval_id)
        return r_item, tag

    def terminate(self):
        logger.info("Closing evaluator process")
        self.eval_init_queue.put(None)
        self.eval_queue.put(None)
        self.evaluator_process.join(timeout=10)
        if self.evaluator_process.exitcode is None:
            logger.warning("Terminating evalautor process")
            # not exited
            self.evaluator_process.terminate()
        else:
            logger.info("Closed evaluator process successfully")
        self.evaluator_process = None


def evaluator_loop(model, config: EvaluatorConfig, eval_queue: multiprocessing.Queue,
                   result_queue: multiprocessing.Queue):
    while True:
        eval_items = eval_queue.get()  # type: List[EvalItem]
        logger.info("Got eval items")
        if eval_items is None:
            break
        dnn_input = np.stack([eval_item.dnn_input for eval_item in eval_items])
        if config.gpu >= 0:
            dnn_input = chainer.cuda.to_gpu(dnn_input)
        model_output_var_move, model_output_var_value = model.forward(dnn_input)
        model_output_var_move = F.softmax(model_output_var_move)
        model_output_var_value = F.tanh(model_output_var_value)
        model_output_var_move = chainer.cuda.to_cpu(model_output_var_move.data)
        model_output_var_value = chainer.cuda.to_cpu(model_output_var_value.data)
        result_items = []  # type: List[ResultItem]
        for i in range(len(eval_items)):
            result_item = ResultItem()
            result_item.eval_id = eval_items[i].eval_id
            result_item.score = model_output_var_value[i]
            result_item.move_probs = model_output_var_move[i]
            result_items.append(result_item)
        logger.info("Sending result")
        result_queue.put(result_items)
    logger.info("Stopping evaluator")


def evaluator_main(init_queue: multiprocessing.Queue, init_complete_event: multiprocessing.Event,
                   eval_queue: multiprocessing.Queue, result_queue: multiprocessing.Queue):
    logger.info("Evaluator main")
    init_complete_event.set()
    while True:
        config = init_queue.get()  # type: EvaluatorConfig
        if config is None:
            break
        logger.info("Initializing evaluator")
        model = load_model(config.model_path)
        if config.gpu >= 0:
            chainer.cuda.get_device_from_id(config.gpu).use()
            model.to_gpu()

        init_complete_event.set()
        with chainer.using_config("train", False):
            with chainer.using_config("enable_backprop", False):
                evaluator_loop(model, config, eval_queue, result_queue)
    logger.info("Exiting evaluator")
