"""
YaneuraOu.exeを用いて、packed kifuを読み取り盤面・指し手・評価値を得るデータセットリーダー
"""
import os
import subprocess
import msvcrt
import atexit
from collections import deque
from typing import Iterable

import numpy as np

# 別ディレクトリにコピーして使われるので、相対import不可
from neneshogi import config

BOARD_SHAPES = {1: (61, 9, 9)}


class PackedKifuDataset():
    """
    棋譜を読み取り盤面などを返す。
    ランダムアクセス不可。
    """
    board_shape: Iterable[int]
    board_byte_size: int
    record_byte_size: int

    def __init__(self, kifu_path, n_samples, sample_offset, board_format=1, move_format=1):
        self.board_format = board_format
        self.move_format = move_format
        self.board_shape = BOARD_SHAPES[board_format]
        self.board_byte_size = int(np.prod(self.board_shape) * 4)
        self.record_byte_size = self.board_byte_size + 2 * 4

        self.yaneuraou_engine = subprocess.Popen([config.YANEURAOU_KIFU_DECODER_EXE],
                                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        # stdioをバイナリで使うための設定
        msvcrt.setmode(self.yaneuraou_engine.stdin.fileno(), os.O_BINARY)
        msvcrt.setmode(self.yaneuraou_engine.stdout.fileno(), os.O_BINARY)
        atexit.register(self.close)
        self.yaneuraou_engine.stdin.write(b"usi\r\nisready\r\n")
        self.yaneuraou_engine.stdin.flush()
        while True:
            line = self.yaneuraou_engine.stdout.readline()
            if len(line) == 0:
                raise Exception("Cannot initialize YaneuraOu")
            if line.startswith(b"readyok"):
                break

        self._decode_query = "user decode_train_sfen {} {} {} {} {}\r\n".format(
            board_format, move_format, kifu_path, sample_offset, n_samples).encode()

        self._length = n_samples
        self.data_tuples = deque()
        self._remain_buf = b""
        self._pending_size = 0

    def _show_sample(self, x_data, y_data):
        x_copy = x_data[:29, :, :].copy()
        x_copy[28, :, :] = 0.5
        am = np.argmax(x_copy, axis=0)
        am[am == 28] = -1
        print(am)
        y_piece = y_data // 81
        y_y = y_data // 9 % 9
        y_x = y_data % 9
        d = np.zeros((9, 9), dtype=np.int32)
        d[:] = -1
        d[y_y, y_x] = y_piece
        print(d)

    def _fill_data_tuples(self):
        if self._pending_size == 0:
            assert len(self._remain_buf) == 0, str(len(self._remain_buf))
            self.yaneuraou_engine.stdin.write(self._decode_query)
            self.yaneuraou_engine.stdin.flush()
            self._pending_size = self._length * self.record_byte_size

        if len(self._remain_buf) < self.record_byte_size:
            read_data = self.yaneuraou_engine.stdout.read(min(self._pending_size, 65536))
            if len(read_data) == 0:
                raise Exception("Cannot read from decoder")
            self._remain_buf += read_data
            self._pending_size -= len(read_data)
        n_available = len(self._remain_buf) // self.record_byte_size
        for i in range(n_available):
            byte_ofs = i * self.record_byte_size
            board_data = np.fromstring(self._remain_buf[byte_ofs:byte_ofs + self.board_byte_size],
                                       dtype=np.float32).reshape(self.board_shape)
            y_data_array = np.fromstring(self._remain_buf[
                                         byte_ofs + self.board_byte_size:byte_ofs + self.record_byte_size],
                                         dtype=np.int32)
            # 盤面(float32)、指し手(int32)、評価値(int32)
            self.data_tuples.append((board_data, y_data_array[0], y_data_array[1]))
            # self._show_sample(x_data, y_data)
        self._remain_buf = self._remain_buf[n_available * self.record_byte_size:]

    def __getitem__(self, index):
        if isinstance(index, slice):
            index_indices = index.indices(self._length)  # (start, stop, step) without None
            count = index_indices[1] - index_indices[0]
            while len(self.data_tuples) < count:
                self._fill_data_tuples()
            data = [self.data_tuples.pop() for i in range(count)]
            return data
        else:
            # assume integer
            if len(self.data_tuples) == 0:
                self._fill_data_tuples()
            return self.data_tuples.popleft()

    def __len__(self):
        return self._length

    def close(self):
        if self.yaneuraou_engine is not None:
            self.yaneuraou_engine.terminate()
            self.yaneuraou_engine = None
