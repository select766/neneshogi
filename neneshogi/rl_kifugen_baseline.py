"""
方策強化学習用ベースライン棋譜生成
yaneuraouのsearch(=駒得評価で数手読む)で自己対戦

yaneuraouの40バイトの形式の棋譜を出力する。評価値は意味なし。
ランダム性のため、最初に20手ランダムムーブしてから開始する。この部分は棋譜に含まない。
"""
import os
import sys
import argparse
import random
import struct
from tqdm import tqdm
from .move import Piece, Color, Move
from .position import Position, PositionHelper

pack_fmt = "<hHHbb"  # short score, ushort move, ushort gamePly, signed char result, signed char pad


class RLKifugenBaseline:
    first_random_moves: int
    search_depth: int
    draw_moves: int

    def __init__(self, first_random_moves: int):
        self.first_random_moves = first_random_moves
        self.search_depth = 3
        self.draw_moves = 256

    def _random_move(self, pos: Position):
        for i in range(self.first_random_moves):
            m = random.choice(pos.generate_move_list())
            pos.do_move(m)

    def play(self, f) -> int:
        """
        1局行って棋譜をファイルに書き出す。
        :param f: ファイルオブジェクト
        :return: 書き出された局面数。引き分けでは0。
        """
        pos = Position()
        pos.set_hirate()
        self._random_move(pos)

        sfens = []  # packed sfen
        moves = []  # int
        side_to_move_list = []
        game_ply_list = []
        winner = 0
        while pos.game_ply() <= self.draw_moves:  # game_ply()は初形で1
            m = pos.search(self.search_depth)
            if m == Move.MOVE_RESIGN:
                winner = 1 - pos.side_to_move()
                break
            sfens.append(pos.sfen_pack())
            moves.append(m.to_int())
            side_to_move_list.append(pos.side_to_move())
            game_ply_list.append(pos.game_ply())
            pos.do_move(m)
        else:
            # 引き分け
            # 出力しない
            return 0
        for i in range(len(sfens)):
            game_result = 1 if winner == side_to_move_list[i] else -1
            f.write(sfens[i])
            f.write(struct.pack(pack_fmt, 0, moves[i], game_ply_list[i], game_result, 0))
        return len(sfens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dst")
    parser.add_argument("n_positions", type=int)
    parser.add_argument("--first_random_moves", type=int, default=20)
    parser.add_argument("--search_depth", type=int, default=3)
    args = parser.parse_args()
    gen = RLKifugenBaseline(args.first_random_moves)
    gen.search_depth = args.search_depth
    completed_positions = 0
    pbar = tqdm(total=args.n_positions)
    with open(args.dst, "wb") as f:
        while completed_positions < args.n_positions:
            n_game_pos = gen.play(f)
            completed_positions += n_game_pos
            pbar.update(n_game_pos)
    pbar.close()


if __name__ == "__main__":
    main()
