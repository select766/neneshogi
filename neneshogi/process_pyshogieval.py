"""
pyshogievalによる評価リクエストを処理するプロセス
"""
from typing import Dict, Optional, List
import argparse

from logging import getLogger

logger = getLogger(__name__)

import numpy as np
import chainer

from pyshogieval import ShogiEval
from .train_config import load_model
from . import util


def run(seval: ShogiEval, batch_size: int, model, gpu: int, softmax_temperature: float, value_slope: float):
    dnn_input_batch = np.zeros((batch_size, ShogiEval.DNN_INPUT_CHANNEL, 9, 9), dtype=np.float32)
    dnn_move_and_index = np.zeros((batch_size, ShogiEval.MOVE_SIZE, 2), dtype=np.uint16)
    n_moves = np.zeros((batch_size,), dtype=np.uint16)
    if True:
        logger.info("warming up by dummy data")
        if gpu >= 0:
            dnn_input_gpu = chainer.cuda.to_gpu(dnn_input_batch)
        else:
            dnn_input_gpu = dnn_input_batch
        model_output_var_move, model_output_var_value = model.forward(dnn_input_gpu)
        model_output_move = chainer.cuda.to_cpu(model_output_var_move.data)
        model_output_value = chainer.cuda.to_cpu(model_output_var_value.data)
        logger.info("warming up done")

    while True:
        logger.info("waiting input")
        valid_batch_size, table = seval.get(dnn_input_batch, dnn_move_and_index, n_moves)
        logger.info(f"evaluating bs={valid_batch_size}")
        if gpu >= 0:
            dnn_input_gpu = chainer.cuda.to_gpu(dnn_input_batch[:valid_batch_size])
        else:
            dnn_input_gpu = dnn_input_batch[:valid_batch_size]
        model_output_var_move, model_output_var_value = model.forward(dnn_input_gpu)
        model_output_move = chainer.cuda.to_cpu(model_output_var_move.data)
        model_output_value = chainer.cuda.to_cpu(model_output_var_value.data)
        move_and_prob = np.zeros((batch_size, ShogiEval.MOVE_SIZE, 2), dtype=np.uint16)
        for b in range(valid_batch_size):
            n_moves_b = int(n_moves[b])
            if n_moves_b > 0:
                dnn_move_and_index_valid = dnn_move_and_index[b, :n_moves_b, :]
                target_move_scores = model_output_move[b][dnn_move_and_index_valid[:, 1]]
                ms_exp = np.exp((target_move_scores - np.max(target_move_scores)) / softmax_temperature)
                move_probs = ms_exp / np.sum(ms_exp)
                move_probs_uint16 = (move_probs * 65535).astype(np.uint16)
                move_and_prob[b, :n_moves_b, 0] = dnn_move_and_index_valid[:, 0]
                move_and_prob[b, :n_moves_b, 1] = move_probs_uint16
        static_value_int16 = (np.tanh(model_output_value * value_slope) * 32000).astype(np.int16)
        logger.info("sending back")
        seval.put(valid_batch_size, table, move_and_prob, n_moves, static_value_int16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--queue_prefix", default="neneshogi")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--softmax", type=float, default=1.0)
    parser.add_argument("--value_slope", type=float, default=1.0)
    args = parser.parse_args()
    model = load_model(args.model)
    gpu = args.gpu
    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()
    queue_size = 64
    batch_size = args.batch_size
    seval = ShogiEval(queue_size, batch_size, args.queue_prefix)
    with chainer.using_config("train", False):
        with chainer.using_config("enable_backprop", False):
            run(seval, batch_size, model, gpu, args.softmax, args.value_slope)


if __name__ == '__main__':
    try:
        main()
    except Exception as ex:
        logger.exception("fatal error")
