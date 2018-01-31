"""
yaneuraou moduleを用いて、packed kifuを読み取り盤面・指し手・評価値・勝敗を得るデータセットリーダー
"""
from typing import Iterable
from yaneuraou import Position, Move, DNNConverter

import numpy as np

# 別ディレクトリにコピーして使われるので、相対import不可
# from neneshogi import config


class PackedKifuDataset():
    """
    棋譜を読み取り盤面などを返す。
    ランダムアクセス不可。
    """
    board_shape: Iterable[int]
    record_byte_size: int
    move_shape: Iterable[int]
    pos: Position
    dnn_converter: DNNConverter
    packed_array: np.ndarray

    def __init__(self, kifu_path, n_samples, sample_offset, board_format=1, move_format=1):
        self.board_format = board_format
        self.move_format = move_format
        self.pos = Position()
        self.dnn_converter = DNNConverter(board_format, move_format)

        self.board_shape = self.dnn_converter.board_shape()
        self.move_shape = self.dnn_converter.move_shape()
        self.record_byte_size = 40

        self._length = n_samples
        self.packed_array = np.memmap(kifu_path, dtype=np.uint8, mode="r",
                                      offset=self.record_byte_size * sample_offset,
                                      shape=(self._length, 40))

    def _decode_one(self, packed_sample):
        pos = self.pos
        pos.set_from_packed_sfen_value(packed_sample)
        pv_move = pos.psv_move  # type: Move
        board_array = self.dnn_converter.get_board_array(pos)
        move_index = self.dnn_converter.get_move_index(pos, pv_move)
        move_array = self.dnn_converter.get_move_array(pos, pv_move)
        legal_move_array = self.dnn_converter.get_legal_move_array(pos)
        raw_score = pos.psv_score
        eval_score = np.tanh(raw_score / 1200.0).astype(np.float32)
        game_result = np.array(pos.psv_game_result, dtype=np.float32)
        return (board_array, move_index, move_array, legal_move_array, eval_score, game_result)

    def __getitem__(self, index):
        if isinstance(index, slice):
            data = [self._decode_one(self.packed_array[i]) for i in range(*index.indices(self._length))]
            return data
        else:
            # assume integer
            return self._decode_one(self.packed_array[index])

    def __len__(self):
        return self._length
