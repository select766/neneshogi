"""
モデルにダミーデータを流し、GPU性能のベンチマークをする。
"""

from typing import Dict, Optional, List
import argparse
import time

import numpy as np
import chainer

from pyshogieval import ShogiEval
from ..train_config import load_model
from .. import util


def run(batch_size: int, model, gpu: int, test_time: int):
    dnn_input_batch = np.random.random((batch_size, ShogiEval.DNN_INPUT_CHANNEL, 9, 9)).astype(np.float32)

    begin_time = 0
    end_time = time.time() + test_time

    batch_count = -1
    while True:
        if gpu >= 0:
            dnn_input_gpu = chainer.cuda.to_gpu(dnn_input_batch)
        else:
            dnn_input_gpu = dnn_input_batch
        model_output_var_move, model_output_var_value = model.forward(dnn_input_gpu)
        model_output_move = chainer.cuda.to_cpu(model_output_var_move.data)
        model_output_value = chainer.cuda.to_cpu(model_output_var_value.data)
        batch_count += 1

        now_time = time.time()
        if begin_time == 0:
            print("started measurement")
            begin_time = now_time
        else:
            if now_time >= end_time:
                end_time = now_time
                break

    samples_per_sec = batch_count * batch_size / (end_time - begin_time)
    print(f"batch_size={batch_size}, samples_per_sec={samples_per_sec}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--time", type=int, default=30)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_cudnn", default="auto")
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument("--use_cudnn_tensor_core", default="auto")
    args = parser.parse_args()
    model = load_model(args.model)
    gpu = args.gpu
    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()
    batch_size = args.batch_size
    chainer.config.train = False
    chainer.config.enable_backprop = False
    chainer.config.use_cudnn = args.use_cudnn
    chainer.config.autotune = args.autotune
    chainer.config.use_cudnn_tensor_core = args.use_cudnn_tensor_core
    run(batch_size, model, gpu, args.time)


if __name__ == '__main__':
    try:
        main()
    except Exception as ex:
        print(f"fatal error {ex}")
