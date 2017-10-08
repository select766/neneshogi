"""
定跡にかかわる処理

やねうら王の定跡フォーマットを利用する。引用:
https://github.com/yaneurao/YaneuraOu/blob/891b8afee640d54699e3c96598528e60d38d9936/docs/%E8%A7%A3%E8%AA%AC.txt
    [定跡フォーマット]
    sfen sfen文字列
    この局面での指し手1 相手の応手1 この指し手を指したときの評価値 そのときの探索深さ その指し手が選択された回数
    この局面での指し手2 相手の応手2 この指し手を指したときの評価値 そのときの探索深さ その指し手が選択された回数
    …
    sfen xxxx..
    相手の応手がないときは"none"と書いておく。

    例)
    sfen lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1
    7g7f 3c3d 0 32 2
    sfen lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 2
    3c3d 2g2f 0 32 1

局面からsfenを生成する機能はPositionクラスにあるので、単純にSFENをキーとしたdictを用いる。
"""
import random
from collections import defaultdict
from typing import Optional, List, Dict

from logging import getLogger

logger = getLogger(__name__)
from .position import Position, Move
from .usi_info_writer import UsiInfoWriter


class BookMove:
    move: Move
    counter_move: Move
    value: int
    depth: int
    count: int
    probability: float

    def __init__(self, line: str):
        parts = line.split()
        self.move = Move.from_usi_string(parts[0])
        if parts[1] != "none":
            self.counter_move = Move.from_usi_string(parts[1])
        else:
            self.counter_move = None
        self.value = int(parts[2])
        self.depth = int(parts[3])
        self.count = int(parts[4])
        self.probability = 0.0

    @staticmethod
    def assign_probability(all_book_moves: List["BookMove"]):
        total_count = 0
        for bm in all_book_moves:
            total_count += bm.count
        if total_count == 0:
            total_count = 1
        for bm in all_book_moves:
            bm.probability = bm.count / total_count

    def write_pv(self, usi_info_writer: UsiInfoWriter):
        pv = [self.move]
        if self.counter_move is not None:
            pv.append(self.counter_move)
        usi_info_writer.write_pv(pv=pv, depth=self.depth, nodes=0,
                                 score_cp=self.value)
        usi_info_writer.write_string(f"{self.move.to_usi_string()} ({int(self.probability*100)}%)")


class Book:
    sfen_to_move: Dict[str, List[BookMove]]

    def __init__(self):
        self.sfen_to_move = defaultdict(list)

    def load(self, path: str):
        current_sfen = None
        with open(path) as f:
            for line in f:
                line_s = line.rstrip()
                if line_s.startswith("#"):
                    continue
                elif line_s.startswith("sfen "):
                    current_sfen = line_s[5:]
                elif len(line_s) > 0:
                    # move
                    bm = BookMove(line_s)
                    self.sfen_to_move[current_sfen].append(bm)
        for bm_list in self.sfen_to_move.values():
            BookMove.assign_probability(bm_list)

    def get_move(self, pos: Position) -> Optional[BookMove]:
        sfen = pos.get_sfen()
        logger.info(f"Searching book for {sfen}")
        if sfen in self.sfen_to_move:
            bm_list = self.sfen_to_move[sfen]
            if len(bm_list) > 0:
                # 重みづけchoice
                weights = [bm.probability for bm in bm_list]
                return random.choices(bm_list, weights=weights)[0]
            else:
                return bm_list[0]
        else:
            return None
