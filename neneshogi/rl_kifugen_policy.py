"""
DNN policyに従って自己対戦・棋譜生成をする。
同時にバッチサイズ分対局を進めて、一括でGPU評価する。
"""

import os
import sys
import argparse
import random
import struct

import numpy as np
import chainer
from tqdm import tqdm
from typing import List

from .move import Piece, Color, Move
from yaneuraou import DNNConverter
from .position import Position, PositionHelper
from .train_config import load_model


class KifuWriter:
    pack_fmt = "<hHHbb"  # short score, ushort move, ushort gamePly, signed char result, signed char pad
    count: int

    def __init__(self, path: str):
        self.f = open(path, "wb")
        self.count = 0

    def write(self, sfen: bytes, score: int, move: int, game_ply: int, game_result: int):
        self.f.write(sfen + struct.pack(KifuWriter.pack_fmt, score, move, game_ply, game_result, 0))
        self.count += 1

    def close(self):
        self.f.close()


class RLKifugenFiber:
    first_random_moves: int
    draw_moves: int
    pos: Position
    next_move: Move
    kifu_writer: KifuWriter

    def __init__(self):
        self.first_random_moves = 20
        self.draw_moves = 256

    def generate_position(self, kifu_writer: KifuWriter):
        self.kifu_writer = kifu_writer
        while True:
            for dummy in self._play_once():
                yield dummy

    def _random_move(self, pos: Position):
        for i in range(self.first_random_moves):
            m = random.choice(pos.generate_move_list())
            pos.do_move(m)

    def _play_once(self):
        pos = Position()
        self.pos = pos
        pos.set_hirate()

        sfens = []  # packed sfen
        moves = []  # int
        side_to_move_list = []
        game_ply_list = []
        winner = 0
        while pos.game_ply() <= self.draw_moves:  # game_ply()は初形で1
            if pos.is_mated():
                winner = 1 - pos.side_to_move()
                break
            yield None
            m = self.next_move
            sfens.append(pos.sfen_pack())
            moves.append(m.to_int())
            side_to_move_list.append(pos.side_to_move())
            game_ply_list.append(pos.game_ply())
            pos.do_move(m)
        else:
            # 引き分け
            # 出力しない
            return
        for i in range(len(sfens)):
            game_result = 1 if winner == side_to_move_list[i] else -1
            self.kifu_writer.write(sfens[i], 0, moves[i], game_ply_list[i], game_result)


class RLKifugenPolicy:
    batch_size: int
    models: List[chainer.Chain]
    gpu: int
    dnn_converter: DNNConverter
    softmax_temperature: float

    def __init__(self):
        self.dnn_converter = DNNConverter(1, 1)

    def run(self, path: str, n_positions: int):
        kifu_writer = KifuWriter(path)
        fibers = [RLKifugenFiber() for _ in range(self.batch_size)]
        iters = [fiber.generate_position(kifu_writer) for fiber in fibers]
        model_idx = 0
        pbar = tqdm(total=n_positions)
        last_count = 0
        while kifu_writer.count < n_positions:
            for it in iters:
                next(it)  # yieldまで進める

            dnn_input = np.stack([self.dnn_converter.get_board_array(fiber.pos) for fiber in fibers])
            legal_move_mask = np.stack([self.dnn_converter.get_legal_move_array(fiber.pos) for fiber in fibers])
            with chainer.using_config("train", False):
                if self.gpu >= 0:
                    dnn_input = chainer.cuda.to_gpu(dnn_input)
                model_output_var_move, model_output_var_value = self.models[model_idx].forward(dnn_input)
                model_output = chainer.cuda.to_cpu(model_output_var_move.data)  # type: np.ndarray
            # softmaxで確率とみなす（合法手の和=1）
            mo_exp = np.exp((model_output - np.max(model_output, axis=1,
                                                   keepdims=True)) * self.softmax_temperature) * legal_move_mask.reshape(
                (self.batch_size, -1))
            model_output = mo_exp / np.sum(mo_exp, axis=1, keepdims=True)
            # 確率にしたがって手を選択
            for i, fiber in enumerate(fibers):
                move_index = np.random.choice(np.arange(model_output.shape[1]), p=model_output[i])
                move = self.dnn_converter.reverse_move_index(fiber.pos, move_index)
                fiber.next_move = move
            model_idx = (model_idx + 1) % len(self.models)

            pbar.update(kifu_writer.count - last_count)
            last_count = kifu_writer.count
        pbar.close()
        kifu_writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dst")
    parser.add_argument("n_positions", type=int)
    parser.add_argument("model1")
    parser.add_argument("model2")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu)
    models = []
    for model_path in [args.model1, args.model2]:
        model = load_model(model_path)
        if args.gpu >= 0:
            model.to_gpu()
        models.append(model)

    gen = RLKifugenPolicy()
    gen.batch_size = args.batch_size
    gen.gpu = args.gpu
    gen.softmax_temperature = 0.1
    gen.models = models
    gen.run(args.dst, args.n_positions)


if __name__ == "__main__":
    main()
