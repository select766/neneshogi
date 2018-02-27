"""
MCTSによる自己対戦・棋譜生成
自己対戦を一定数行い、学習データを生成する。
局面ごとに
- 局面のsfen
- 探索結果の確率分布
- 手番側の勝敗
を収集。
複数プロセスを同時に実行し、GPUを使い切りつつ処理する。
今のところ1つのモデルのみ使用可能。

"""
import os
import sys
import queue
from typing import List, Dict
import gzip
import pickle
import uuid
import time
import datetime
import multiprocessing
import numpy as np
from .usi_info_writer import UsiInfoWriter
from .position import Position, Color, Square, Piece, Move, PositionHelper

from .mcts_player import MCTSPlayer


class MCTSKifuGen:
    def __init__(self):
        pass


class MCTSKifuGenProcess:
    """
    自己対戦を行う1プロセス。
    """
    engines: List[MCTSPlayer]
    record_dir: str
    engine_options_list: List[Dict[str, str]]
    exit_event: multiprocessing.Event
    complete_queue: multiprocessing.Queue

    def __init__(self, record_dir: str, engine_options_list: List[Dict[str, str]], exit_event: multiprocessing.Event,
                 complete_queue: multiprocessing.Queue):
        self.record_dir = record_dir
        self.exit_event = exit_event
        self.complete_queue = complete_queue
        self.engines = []
        for i in range(2):
            engine = MCTSPlayer(evaluator=None, kifu_gen=True)
            self.engines.append(engine)
        self.engine_options_list = engine_options_list

    def run(self):
        black_engine = 0
        while not self.exit_event.is_set():
            match_uuid = str(uuid.uuid4())
            records = self._single_match(black_engine)
            self._serialize_match_to_file(os.path.join(self.record_dir, match_uuid + ".pickle.gz"), match_uuid,
                                          black_engine, records)
            self.complete_queue.put((match_uuid, 1, len(records)))
            black_engine = 1 - black_engine
            time.sleep(1)  # 今回の対局で必要数集まったなら終了イベントを設定してほしいのでその分待機

    def _serialize_match_to_file(self, path, match_uuid, black_engine, records):
        metadata = {"uuid": match_uuid, "datetime": datetime.datetime.utcnow(),
                    "engine_options_list": self.engine_options_list, "black_engine": black_engine}
        with gzip.open(path, "wb") as f:
            pickle.dump(metadata, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(records, f, pickle.HIGHEST_PROTOCOL)

    def _single_match(self, black_engine: int):
        current_engine = black_engine
        pos = Position()
        pos.set_hirate()
        dummy_uiw = UsiInfoWriter(lambda x: None)
        for i in range(2):
            eng = self.engines[i]
            eng.isready(self.engine_options_list[i])
            eng.usinewgame()
        sfens = []
        moves = []
        plys = []  # 初手が1
        scores = []
        legal_moves = []
        visits = []
        game_result = 0
        for ply in range(1, 256 + 1):
            eng = self.engines[current_engine]
            eng.pos = pos
            bestmove, score, legal_move, visit = eng._make_strategy(dummy_uiw)
            if bestmove == Move.MOVE_RESIGN:
                # 投了
                # 探索結果自体は保存しても仕方ない

                # たとえば初手投了だと、ply%2==1で、後手の勝ち。
                game_result = 1 if ply % 2 == 0 else -1
                break
            plys.append(ply)
            sfens.append(pos.sfen())
            moves.append(bestmove.to_usi_string())
            scores.append(score)
            legal_move_binary = np.array([m.to_int() for m in legal_move], dtype=np.uint16).tobytes()
            legal_moves.append(legal_move_binary)
            visits.append(visit.astype(np.uint16).tobytes())

            pos.do_move(bestmove)
            current_engine = 1 - current_engine

        for i in range(2):
            eng = self.engines[i]
            eng.gameover("draw")

        game_results = []  # 手番側が勝っていれば1,負けていれば-1,引き分けは0
        for i in range(len(plys)):
            game_results.append(game_result)
            game_result = -game_result

        records = list(zip(sfens, moves, plys, scores, legal_moves, visits, game_results))
        return records


def mcts_kifu_gen_process_main(**kwargs):
    kifu_gen = MCTSKifuGenProcess(**kwargs)
    kifu_gen.run()


def main():
    exit_event = multiprocessing.Event()
    complete_queue = multiprocessing.Queue()
    need_count = 2
    procs = []
    engine_options = {
        "model_path": r"D:\dev\shogi\neneshogi\neneshogi_private\data\model\20180131232553\weight\model_iter_390625",
        "nodes": "1000",
        "c_puct": 1,
        "play_temperature": 0.1,
        "batch_size": 16,
        "gpu": 0}
    for i in range(2):
        proc = multiprocessing.Process(target=mcts_kifu_gen_process_main,
                                       kwargs={"record_dir": "data/rl/data",
                                               "engine_options_list": [engine_options, engine_options],
                                               "exit_event": exit_event, "complete_queue": complete_queue})
        proc.start()
        procs.append(proc)
    while need_count > 0:
        print(complete_queue.get())
        need_count -= 1
    print("need count 0")
    exit_event.set()
    while len(procs) > 0:
        procs[0].join(timeout=1)
        if procs[0].exitcode is not None:
            procs.pop(0)
        try:
            print(complete_queue.get_nowait())
        except queue.Empty:
            pass
    print("all joined")


if __name__ == "__main__":
    main()
