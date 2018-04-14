"""
DBの局面を順番に投入し、定跡を作成する。

- ユニークなワークidを設定(整数)
- エンジン初期化（与えられたoptionコマンド使用）
- 局面を投入し、bestmoveを得る
- bestmoveをdbに保存
- hashfullが規定値に達したら置換表を保存、エンジンを再起動

作業ディレクトリ構成
- book.db: 定跡db(sqlite)
- config.yaml: 設定
- options.txt: isreadyの前に送るコマンド列テキスト
- log: USIエンジンとのやり取りログディレクトリ(ワークidごとに作成)
- ttable: 置換表保存ディレクトリ(ワークidごとにファイル名を決定し作成)

config
- engine_path: エンジンexeのパス
- go: goコマンド(長い思考時間を設定する)
- save_min_visit: 置換表のうち保存する要素の最小訪問回数
"""

import os
import sys
import argparse
import subprocess
import sqlite3
import time
from collections import defaultdict
from logging import getLogger

logger = getLogger(__name__)

from typing import List, Dict, Tuple
from .. import util


class Engine:
    _proc: subprocess.Popen

    def __init__(self, config: Dict, work_dir: str, work_id: int):
        self._log_file = open(os.path.join(work_dir, "log", f"{work_id}.log"), "a")
        self._proc = subprocess.Popen([config["engine_path"]],
                                      stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE,
                                      encoding="utf-8",
                                      cwd=os.path.dirname(config["engine_path"]))
        self._init_engine(config, work_dir)

    def _init_engine(self, config: Dict, work_dir: str):
        self.write_line("usi")
        while True:
            line = self.read_line()
            if line == "usiok":
                break
        with open(os.path.join(work_dir, "options.txt")) as f:
            for line in f:
                self.write_line(line.rstrip())
        self.write_line("isready")
        while True:
            line = self.read_line()
            if line == "readyok":
                break
        self.write_line("usinewgame")

    def write_line(self, line: str):
        self._log_file.write(f">{line}\n")
        self._proc.stdin.write(line + "\n")
        self._proc.stdin.flush()

    def read_line(self):
        line = self._proc.stdout.readline().rstrip()
        self._log_file.write(f"<{line}\n")
        if len(line) == 0:
            raise EOFError
        return line

    def quit(self):
        self.write_line("quit")
        self._proc.wait()
        self._log_file.close()


def init_engine(config: Dict, work_dir: str):
    pass


def load_positions(cur, mincount: int) -> List[Tuple[int, str]]:
    cur.execute("SELECT id, moves FROM book WHERE count >= ? AND best_move IS NULL ORDER BY moves", (mincount,))
    return cur.fetchall()


def consider_one_position(config: Dict, engine: Engine, moves: str) -> Tuple[str, int]:
    engine.write_line(f"position startpos moves {moves}")
    engine.write_line(config["go"])
    hashfull = 0
    while True:
        line = engine.read_line()
        if line.startswith("bestmove"):
            bestmove = line.split(" ")[1]
            return bestmove, hashfull
        else:
            parts = line.split(" ")
            try:
                hashfull_idx = parts.index("hashfull")
                hashfull = int(parts[hashfull_idx + 1])
            except ValueError:
                pass


def do_work(work_dir: str, mincount: int) -> bool:
    config = util.yaml_load(os.path.join(work_dir, "config.yaml"))
    save_min_visit = config["save_min_visit"]
    work_id = int(time.strftime('%Y%m%d%H%M%S'))
    logger.info(f"work id: {work_id}")
    con = sqlite3.connect(os.path.join(work_dir, "book.db"))
    cur = con.cursor()
    id_moves_list = load_positions(cur, mincount)
    logger.info(f"{len(id_moves_list)} positions to consider")
    if len(id_moves_list) == 0:
        return False

    engine = Engine(config, work_dir, work_id)
    for position_id, moves in id_moves_list:
        logger.info(f"Considering position {moves}")
        bestmove, hashfull = consider_one_position(config, engine, moves)
        cur.execute("UPDATE book SET best_move=?, process_id=? WHERE id=?", (bestmove, work_id, position_id))
        con.commit()
        logger.info(f"bestmove {bestmove} hashfull {hashfull}")
        if hashfull >= 900:
            break
    logger.info("saving trans table")
    ttable_path = os.path.join(os.path.abspath(work_dir), "ttable", f"{work_id}.bin")
    engine.write_line(f"user save_tt {save_min_visit} {ttable_path}")
    assert engine.read_line() == "ok"

    logger.info("quitting engine")
    engine.quit()

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("work_dir")
    parser.add_argument("mincount", type=int)

    args = parser.parse_args()
    work_dir = args.work_dir
    os.makedirs(os.path.join(work_dir, "log"), exist_ok=True)
    os.makedirs(os.path.join(work_dir, "ttable"), exist_ok=True)
    while do_work(work_dir, args.mincount):
        pass


if __name__ == '__main__':
    main()
