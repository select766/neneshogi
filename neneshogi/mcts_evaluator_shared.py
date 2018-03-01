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
    gpu_batch_size: int

    def __init__(self, evaluator_config: EvaluatorConfig, n_clients: int, gpu_batch_size: int):
        self.evaluator_config = evaluator_config
        self.n_clients = n_clients
        self._next_client_idx = 0
        self.gpu_batch_size = gpu_batch_size

    def start_process(self):
        logger.info("Starting evaluator process")
        self.init_complete_event = multiprocessing.Event()
        self.eval_queue = multiprocessing.Queue()
        self.result_queues = [multiprocessing.Queue() for i in range(self.n_clients)]
        self.evaluator_process = multiprocessing.Process(target=evaluator_main,
                                                         kwargs={"config": self.evaluator_config,
                                                                 "init_complete_event": self.init_complete_event,
                                                                 "eval_queue": self.eval_queue,
                                                                 "result_queues": self.result_queues,
                                                                 "gpu_batch_size": self.gpu_batch_size})
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


def generate_first_input(eval_queue: multiprocessing.Queue, batch_size: int, dnn_inputs, dnn_input_aux_items):
    eval_items = eval_queue.get()  # type: List[EvalItem]
    item_shape = eval_items[0].dnn_input.shape  # type: Tuple[int,int,int]
    dnn_input = np.zeros((batch_size,) + item_shape, dtype=np.float32)
    for i, eval_item in enumerate(eval_items):
        dnn_input[i] = eval_item.dnn_input
    dnn_inputs.append(dnn_input)
    dnn_input_aux_items.append([eval_items])
    return item_shape


def generate_second_input(eval_queue: multiprocessing.Queue, batch_size: int, item_shape: Tuple[int, int, int],
                          exit_time: float, dnn_inputs, dnn_input_aux_items, last_excess_items: List[EvalItem]) -> List[
    EvalItem]:
    dnn_input = np.zeros((batch_size,) + item_shape, dtype=np.float32)
    ctr = 0
    batch_aux_items = []
    if last_excess_items is not None:
        for eval_item in last_excess_items:
            dnn_input[ctr] = eval_item.dnn_input
            ctr += 1
        batch_aux_items.append(last_excess_items)
    excess_item = None
    while time.time() < exit_time:
        try:
            eval_items = eval_queue.get_nowait()  # type: List[EvalItem]
            if eval_items is None:
                raise StopIteration
            if len(eval_items) > (batch_size - ctr):
                excess_item = eval_items
                break
            for eval_item in eval_items:
                dnn_input[ctr] = eval_item.dnn_input
                ctr += 1
            batch_aux_items.append(eval_items)
        except queue.Empty:
            pass
    logger.info(f"packed {ctr} items, from {len(batch_aux_items)} clients")
    dnn_inputs.append(dnn_input)
    dnn_input_aux_items.append(batch_aux_items)
    return excess_item


def sendback_result(model_output, eval_items_list: List[List[EvalItem]], result_queues: List[multiprocessing.Queue]):
    model_output_var_move, model_output_var_value = model_output

    ctr = 0
    for eval_items in eval_items_list:
        client_idx = eval_items[0].src
        result_items = []  # type: List[ResultItem]
        for i in range(len(eval_items)):
            result_item = ResultItem()
            result_item.eval_id = eval_items[i].eval_id
            result_item.score = model_output_var_value[ctr]
            result_item.move_probs = model_output_var_move[ctr]
            result_items.append(result_item)
            ctr += 1
        logger.info("Sending result")
        result_queues[client_idx].put(result_items)


def evaluator_loop(model, config: EvaluatorConfig, eval_queue: multiprocessing.Queue,
                   result_queues: List[multiprocessing.Queue], gpu_batch_size: int):
    est_gpu_time = 0.1
    dnn_inputs = deque()
    dnn_input_aux_items = deque()
    dnn_outputs = deque()
    item_shape = generate_first_input(eval_queue, gpu_batch_size, dnn_inputs, dnn_input_aux_items)
    excess_item = None

    while True:
        # GPU上の計算を開始
        logger.info("gpu write start")
        gpu_write_start = time.time()
        dnn_input = dnn_inputs.popleft()
        if config.gpu >= 0:
            dnn_input = chainer.cuda.to_gpu(dnn_input)
        model_output_var_move, model_output_var_value = model.forward(dnn_input)
        model_output_var_move = F.softmax(model_output_var_move)
        model_output_var_value = F.tanh(model_output_var_value)

        # 今までに得られた結果を返送
        while len(dnn_outputs) > 0:
            sendback_result(dnn_outputs.popleft(), dnn_input_aux_items.popleft(), result_queues)

        # 次のバッチを生成
        excess_item = generate_second_input(eval_queue, gpu_batch_size, item_shape, gpu_write_start + est_gpu_time, dnn_inputs,
                                            dnn_input_aux_items, excess_item)

        # GPUから結果回収
        gpu_read_start = time.time()
        model_output_var_move = chainer.cuda.to_cpu(model_output_var_move.data)
        model_output_var_value = chainer.cuda.to_cpu(model_output_var_value.data)
        gpu_read_end = time.time()
        gpu_block_time = gpu_read_end - gpu_read_start
        if gpu_block_time < 0.01:
            est_gpu_time = max(est_gpu_time + 0.001, 0.01)
        else:
            est_gpu_time = min(est_gpu_time + 0.001, 1.0)
        logger.info(f"block time: {gpu_block_time}, est_time: {est_gpu_time}")
        dnn_outputs.append((model_output_var_move, model_output_var_value))


def evaluator_main(config: EvaluatorConfig, init_complete_event: multiprocessing.Event,
                   eval_queue: multiprocessing.Queue, result_queues: List[multiprocessing.Queue],
                   gpu_batch_size: int):
    logger.info("Evaluator main")

    logger.info("Initializing evaluator")
    model = load_model(config.model_path)
    if config.gpu >= 0:
        chainer.cuda.get_device_from_id(config.gpu).use()
        model.to_gpu()

    init_complete_event.set()
    try:
        with chainer.using_config("train", False):
            with chainer.using_config("enable_backprop", False):
                evaluator_loop(model, config, eval_queue, result_queues, gpu_batch_size)
    except StopIteration:
        logger.info("Exiting evaluator")
