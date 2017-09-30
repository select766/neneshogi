"""
教師AIと評価対象AIの指し手を比較する。
USIエンジン対応。

エンジンの設定ファイルは、やねうら王自動対局フレームワークのフォーマットに準拠。以下引用:

    1行目にengineの実行ファイル名(同じフォルダに配置)
    2行目に思考時のコマンド
    3行目以降にsetoption等、初期化時に思考エンジンに送りたいコマンドを書く。

  例)
    test.exe
    go btime 100 wtime 100 byoyomi 0
    setoption name Threads value 1
    setoption name Hash value 1024

処理
- 2エンジンで、
  - positionコマンドで詰みでない局面をエンジンに渡す
  - goコマンドで思考
  - bestmoveを取得
- 2エンジンのbestmoveが異なるとき、
  - 各エンジンの指し手を指した後の局面を教師エンジンに渡す
  - goコマンドで試行
  - 最後のinfoコマンドでの評価値を取得(思考開始局面が詰み可能性があることに注意)
"""
import pickle
import sys
import os
import csv
import subprocess
import argparse
from typing import List


class EngineConfig:
    exe_path: str
    go_command: str
    options: List[str]

    def __init__(self, path):
        self.options = []
        with open(path) as f:
            for i, line in enumerate(f):
                l_strip = line.rstrip()
                if i == 0:
                    self.exe_path = l_strip
                elif i == 1:
                    self.go_command = l_strip
                else:
                    self.options.append(l_strip)


class Engine:
    config: EngineConfig
    process: subprocess.Popen

    def __init__(self, config: EngineConfig):
        self.config = config
        engine_process = subprocess.Popen([config.exe_path],
                                          stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                          cwd=os.path.dirname(config.exe_path))
        engine_process.stdin.write(b"usi\r\n")
        for option in config.options:
            engine_process.stdin.write(option.encode("ascii") + b"\r\n")
        engine_process.stdin.write(b"isready\r\n")
        engine_process.stdin.flush()
        while True:
            line = engine_process.stdout.readline()
            print(line)
            if len(line) == 0:
                raise Exception("Cannot initialize Engine")
            if line.startswith(b"readyok"):
                break
        self.process = engine_process

    def close(self):
        if self.process is not None:
            self.process.stdin.write(b"quit\r\n")
            self.process.stdin.flush()
            self.process.wait()
            self.process = None


class EvalOutput:
    bestmove: str
    pv: List[str]
    score: int
    score_type: str

    def __repr__(self):
        s = f"bestmove {self.bestmove}"
        if self.score_type is not None and self.score is not None:
            s += f" score {self.score_type} {self.score}"
        if self.pv is not None:
            s += f" pv {' '.join(self.pv)}"
        return s


def eval_pos(moves: List[str], engine: Engine) -> EvalOutput:
    engine.process.stdin.write(f"position startpos moves {' '.join(moves)}\r\n".encode("ascii"))
    engine.process.stdin.write(engine.config.go_command.encode("ascii") + b"\r\n")
    engine.process.stdin.flush()
    eval_out = EvalOutput()
    while True:
        line = engine.process.stdout.readline().decode("ascii").rstrip()
        parts = line.split(" ")  # type: List[str]
        if len(parts) == 0:
            continue
        cmd = parts.pop(0)
        if cmd == "info":
            while len(parts) > 0:
                subcmd = parts.pop(0)
                if subcmd in ["depth", "seldepth", "time", "nodes", "currmove", "hashfull", "nps"]:
                    parts.pop(0)
                    continue
                elif subcmd == "string":
                    break
                elif subcmd == "pv":
                    eval_out.pv = parts
                    break
                elif subcmd == "score":
                    eval_out.score_type = parts.pop(0)
                    score_str = parts.pop(0)
                    if score_str == "+":  # 詰みだが手数不明の場合
                        score_val = 1
                    elif score_str == "-":
                        score_val = -1
                    else:
                        score_val = int(score_str)
                    eval_out.score = score_val
                    if len(parts) > 0 and parts[0] in ["lowerbound", "upperbound"]:
                        parts.pop(0)
        elif cmd == "bestmove":
            eval_out.bestmove = parts.pop(0)
            break

    return eval_out


class CompareResult:
    moves: List[str]
    teacher_pv: List[str]
    student_pv: List[str]
    teacher_bestmove: str
    student_bestmove: str


def compare_engine(moves: List[str], teacher_engine: Engine, student_engine: Engine) -> dict:
    root_t_eval = eval_pos(moves, teacher_engine)
    root_s_eval = eval_pos(moves, student_engine)
    t_after_t_eval = None
    t_after_s_eval = None
    s_after_t_eval = None
    s_after_s_eval = None
    if root_s_eval.bestmove != root_t_eval.bestmove:
        # 教師・生徒それぞれのbestmoveで局面を進めて、評価値を計算
        t_after_t_eval = eval_pos(moves + [root_t_eval.bestmove], teacher_engine)
        t_after_s_eval = eval_pos(moves + [root_t_eval.bestmove], student_engine)
        s_after_t_eval = eval_pos(moves + [root_s_eval.bestmove], teacher_engine)
        s_after_s_eval = eval_pos(moves + [root_s_eval.bestmove], student_engine)
    return {"moves": moves, "root_t_eval": root_t_eval, "root_s_eval": root_s_eval,
            "t_after_t_eval": t_after_t_eval, "t_after_s_eval": t_after_s_eval,
            "s_after_t_eval": s_after_t_eval, "s_after_s_eval": s_after_s_eval}


def run_compare(args):
    teacher_config = EngineConfig(args.teacher)
    student_config = EngineConfig(args.student)
    teacher_engine = Engine(teacher_config)
    student_engine = Engine(student_config)
    compare_results = []
    with open(args.kifu) as kifu_lines:
        for i, kifu_line in enumerate(kifu_lines):
            if args.games >= 0 and i >= args.games:
                break
            moves = kifu_line.rstrip().split(" ")[2:]  # startpos moves 7g7f ...
            for te in range(args.skipfirst, len(moves) - 1 - args.skiplast):  # 最後の手の後は詰んでいるので使えない
                print(moves[:te])
                compare_results.append(compare_engine(moves[:te], teacher_engine, student_engine))
    teacher_engine.close()
    student_engine.close()
    with open(args.dst, "wb") as f:
        pickle.dump(compare_results, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kifu")
    parser.add_argument("dst")
    parser.add_argument("--teacher", default="teacher.txt")
    parser.add_argument("--student", default="student.txt")
    parser.add_argument("--skipfirst", type=int, default=16)
    parser.add_argument("--skiplast", type=int, default=0)
    parser.add_argument("--games", type=int, default=-1)
    args = parser.parse_args()
    run_compare(args)


if __name__ == "__main__":
    main()
