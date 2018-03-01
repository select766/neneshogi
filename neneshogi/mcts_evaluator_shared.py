"""
棋譜生成用に、複数プロセスから共有するEvaluator
まず評価プロセスを生成し、そこへアクセスするEvaluatorを複数生成する
"""

from collections import deque
from typing import Dict, Optional, List, Tuple
import queue
import multiprocessing

from logging import getLogger

import time

logger = getLogger(__name__)

import numpy as np
import chainer
import chainer.functions as F

from .train_config import load_model

from .mcts_evaluator import EvaluatorBase, EvaluatorConfig, EvalItem, ResultItem, EvaluatorFrame


class DNNServer:
    evaluator_config: EvaluatorConfig
    n_clients: int
    _next_client_idx: int
    init_complete_event: multiprocessing.Event
    eval_queue: multiprocessing.Queue
    result_queues: List[multiprocessing.Queue]
    evaluator_process: multiprocessing.Process

    def __init__(self, evaluator_config: EvaluatorConfig, n_clients: int):
        self.evaluator_config = evaluator_config
        self.n_clients = n_clients
        self._next_client_idx = 0

    def start_process(self):
        logger.info("Starting evaluator process")
        self.init_complete_event = multiprocessing.Event()
        self.eval_queue = multiprocessing.Queue()
        self.result_queues = [multiprocessing.Queue() for i in range(self.n_clients)]
        self.evaluator_process = multiprocessing.Process(target=evaluator_main,
                                                         kwargs={"config": self.evaluator_config,
                                                                 "init_complete_event": self.init_complete_event,
                                                                 "eval_queue": self.eval_queue,
                                                                 "result_queues": self.result_queues})
        self.evaluator_process.start()
        self.init_complete_event.wait()
        logger.info("Evaluator process started")

    def get_evaluator(self) -> "EvaluatorShared":
        assert self._next_client_idx < self.n_clients
        ev = EvaluatorShared(self._next_client_idx, self.eval_queue, self.result_queues[self._next_client_idx])
        self._next_client_idx += 1
        return ev

    def terminate_process(self):
        logger.info("Waiting evaluator process to exit")
        self.eval_queue.put(None)
        self.evaluator_process.join(timeout=10)
        if self.evaluator_process.exitcode is None:
            logger.info("Terminating evaluator process")
            self.evaluator_process.terminate()
        logger.info("Evaluator process exited")


class EvaluatorShared(EvaluatorFrame):
    client_idx: int

    def __init__(self, client_idx: int, eval_queue: multiprocessing.Queue, result_queue: multiprocessing.Queue):
        super().__init__(eval_queue, result_queue)
        self.client_idx = client_idx

    def start(self):
        pass

    def set_config(self, evaluator_config: EvaluatorConfig):
        pass

    def put(self, eval_item: EvalItem, tag: object):
        eval_item.src = self.client_idx
        super().put(eval_item, tag)

    def terminate(self):
        pass


def evaluator_loop(model, config: EvaluatorConfig, eval_queue: multiprocessing.Queue,
                   result_queues: List[multiprocessing.Queue]):
    while True:
        eval_items = eval_queue.get()  # type: List[EvalItem]
        logger.info("Got eval items")
        if eval_items is None:
            break
        dnn_input = np.stack([eval_item.dnn_input for eval_item in eval_items])
        client_idx = eval_items[0].src
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
        result_queues[client_idx].put(result_items)
    logger.info("Stopping evaluator")


def evaluator_main(config: EvaluatorConfig, init_complete_event: multiprocessing.Event,
                   eval_queue: multiprocessing.Queue, result_queues: List[multiprocessing.Queue]):
    logger.info("Evaluator main")

    logger.info("Initializing evaluator")
    model = load_model(config.model_path)
    if config.gpu >= 0:
        chainer.cuda.get_device_from_id(config.gpu).use()
        model.to_gpu()

    init_complete_event.set()
    with chainer.using_config("train", False):
        with chainer.using_config("enable_backprop", False):
            evaluator_loop(model, config, eval_queue, result_queues)
    logger.info("Exiting evaluator")
