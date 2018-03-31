"""
自動自己対戦プログラム
ルールおよびエンジンの実行設定2つを受け取り対局させ、結果をログに出力する。
neneshogi自体には依存しない。合法手の判定等はpython-shogiモジュールを活用。
指し手に時間がかかりすぎている場合はハングアップとみなして強制終了する。

python -m neneshogi.auto_match rule.yaml engine1.yaml engine2.yaml
"""

import os
import sys
import argparse
import subprocess
import re
from collections import OrderedDict
from typing import Dict, List
import time
import threading
import yaml

import shogi
from .auto_match_objects import Rule, EngineConfig, MatchResult, AutoMatchResult


def yaml_load(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f)


def yaml_dump(obj: object, path: str):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(obj, f, default_flow_style=False)


def yaml_safe_dump(obj: object, path: str):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, default_flow_style=False)


class AutoMatch:
    rule: Rule
    engine_config_list: List[EngineConfig]
    engine_handles: List[subprocess.Popen]
    watchdog: threading.Timer

    def __init__(self, rule: Rule, engine_config_list: List[EngineConfig]):
        self.rule = rule
        assert len(engine_config_list) == 2
        self.engine_config_list = engine_config_list
        self._log_file = None
        self.watchdog = None

    def _log(self, msg: str):
        if self._log_file is not None:
            self._log_file.write(f"{time.strftime('%Y%m%d%H%M%S')}:{msg}\n")
            self._log_file.flush()

    def _exec_engine(self, engine_config: EngineConfig) -> subprocess.Popen:
        env_new = os.environ.copy()
        env_new.update(engine_config.env)
        proc = subprocess.Popen(engine_config.path, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                encoding="utf-8",
                                cwd=os.path.dirname(engine_config.path),
                                env=env_new)
        if self.rule.priority_low:
            if os.name == "nt":
                # https://stackoverflow.com/questions/1023038/change-process-priority-in-python-cross-platform
                import win32api, win32process, win32con
                handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, proc.pid)
                win32process.SetPriorityClass(handle, win32process.BELOW_NORMAL_PRIORITY_CLASS)
        return proc

    def _engine_write(self, engine_idx: int, msg: str):
        self._log(f">{engine_idx}:{msg}")
        self.engine_handles[engine_idx].stdin.write(msg + "\n")
        self.engine_handles[engine_idx].stdin.flush()

    def _engine_read(self, engine_idx: int) -> str:
        msg = self.engine_handles[engine_idx].stdout.readline().rstrip()
        self._log(f"<{engine_idx}:{msg}")
        if len(msg) == 0:
            self._log("EOF detected: aborting")
            sys.exit(1)
        return msg

    def _init_engine(self, engine_idx: int, engine_config: EngineConfig):
        self._engine_write(engine_idx, "usi")

        option_values = OrderedDict()
        while True:
            line = self._engine_read(engine_idx)
            if line.startswith("option "):
                m = re.match("option name ([^ ]+) .* default ([^ ]+)", line)
                if m is None:
                    raise ValueError(f"Unexpected USI response {line}")
                option_values[m.group(1)] = m.group(2)
            elif line.startswith("usiok"):
                break
            elif line.startswith("id "):
                pass
            else:
                raise ValueError(f"Unexpected USI response {line}")
        option_values.update(engine_config.options)
        for option_name, option_value in option_values.items():
            self._engine_write(engine_idx, f"setoption name {option_name} value {option_value}")

    def _isready_engine(self, engine_idx: int):
        self._engine_write(engine_idx, "isready")
        while True:
            line = self._engine_read(engine_idx)
            if line.startswith("readyok"):
                break

    def _quit_engine(self, engine_idx: int):
        if self.engine_handles[engine_idx] is None:
            return
        self._engine_write(engine_idx, "quit")
        try:
            self.engine_handles[engine_idx].wait(timeout=5000)
        except subprocess.TimeoutExpired:
            self.engine_handles[engine_idx].terminate()
        finally:
            self.engine_handles[engine_idx] = None

    def _get_bestmove(self, board: shogi.Board, engine_idx: int) -> str:
        position_command = "position startpos"
        if board.move_number > 1:
            position_command += " moves "
            position_command += " ".join(map(str, board.move_stack))  # 7g7f 3c3d ...
        go_command = self.engine_config_list[engine_idx].go
        self.watchdog = threading.Timer(self.rule.max_go_time, self._go_timeout)
        self.watchdog.start()
        self._engine_write(engine_idx, position_command)
        self._engine_write(engine_idx, go_command)
        move = None
        while True:
            line = self._engine_read(engine_idx)
            if line.startswith("bestmove "):
                move = line.split(" ")[1]
                break
        self.watchdog.cancel()
        self.watchdog = None
        return move

    def _go_timeout(self):
        self._log("go command timeout!")
        for handle in self.engine_handles:
            handle.terminate()  # readがEOFを起こすはず
        os._exit(1)

    def _check_repetition_with_check(self, board: shogi.Board) -> bool:
        """
        千日手が成立したときに、連続王手の千日手かどうか調べる
        :param board:
        :return:
        """

        # 現在の局面から戻していって、同じ局面が出るごとにカウンタをデクリメント。
        # 0になるまで連続王手であれば、連続王手の千日手。
        # Cf: 最終局面と同じ局面かつ王手
        # C: 王手
        # N: 王手でない
        # *: なんでもよい
        # ... Cf * C * Cf * Cf * Cf => True
        # ... Cf * N * Cf * Cf * Cf => False
        rep_count = 4
        final_hash = board.zobrist_hash()
        check_if_check = True  # 最後の手番側のときだけ王手チェックをする
        pop_moves = []
        is_true = False
        while board.move_number > 1:
            if board.zobrist_hash() == final_hash:
                rep_count -= 1
                if rep_count == 0:
                    is_true = True
                    break
            if check_if_check:
                if not board.is_check():
                    break
            check_if_check = not check_if_check
            pop_moves.append(board.pop())

        # 局面を元に戻す
        while len(pop_moves) > 0:
            board.push(pop_moves.pop())
        return is_true

    def _run_single_match(self, black_engine: int) -> MatchResult:
        for i in range(2):
            # ゲームごとにisreadyが必要。
            self._isready_engine(i)
        for i in range(2):
            self._engine_write(i, "usinewgame")
        current_engine = black_engine
        board = shogi.Board()
        draw = False
        winner = None
        gameover_reason = ""
        while True:
            bestmove = self._get_bestmove(board, current_engine)
            if bestmove == "resign":
                winner = 1 - current_engine
                gameover_reason = "resign"
                break
            if bestmove == "win":
                # 宣言勝ち
                # 審査せずに勝ちとみなす
                winner = current_engine
                gameover_reason = "win"
                break
            move_obj = shogi.Move.from_usi(bestmove)
            if move_obj not in board.generate_legal_moves():
                # board.is_legal()は、相手陣から引くときに成る場合がなぜかillegalとなってしまう
                # 非合法手(連続王手の千日手は判定されない)
                winner = 1 - current_engine
                gameover_reason = "illegal_move"
                break
            board.push(move_obj)
            if board.is_fourfold_repetition():
                # 千日手
                if self._check_repetition_with_check(board):
                    # 連続王手の千日手
                    # 最後の指し手が非合法なので、負け
                    winner = 1 - current_engine
                    gameover_reason = "repetition_with_check"
                else:
                    gameover_reason = "repetition"
                    draw = True
                break
            if board.move_number > self.rule.max_moves:
                # 手数制限により引き分け
                # 初形がmove_number==1なので、256手で引き分けなら257で判定
                # 最大手数で詰んでいるときは、例外的に最後の手を指した側の勝ち(stalemateは微妙だが)
                if board.is_game_over():
                    winner = current_engine
                else:
                    draw = True
                gameover_reason = "max_moves"
                break
            current_engine = 1 - current_engine

        for i in range(2):
            win_code = "draw" if draw else ("win" if i == winner else "lose")
            self._engine_write(i, f"gameover {win_code}")

        kifu = list(map(str, board.move_stack))
        return MatchResult(draw, winner, gameover_reason, kifu)

    def _writeout_game_results(self, path: str, match_results: List[MatchResult]):
        data = AutoMatchResult()
        data.rule = self.rule
        data.engine_config_list = self.engine_config_list
        data.match_results = match_results
        yaml_dump(data, path)

    def run_matches(self, log_prefix: str):
        self.engine_handles = []
        match_results = []
        self._log_file = open(log_prefix + ".log", "a")
        try:
            for i, ec in enumerate(self.engine_config_list):
                self._log(f"Initializing engine {i}")
                self.engine_handles.append(self._exec_engine(ec))
                self._init_engine(i, ec)
            for match in range(self.rule.n_match):
                self._log("Start match " + str(match))
                match_result = self._run_single_match(match % 2)
                match_results.append(match_result)
                self._writeout_game_results(log_prefix + ".yaml", match_results)
            for i in range(len(self.engine_config_list)):
                self._log(f"Closing engine {i}")
                self._quit_engine(i)
            self._log("Finished task")
        except Exception as ex:
            self._log(f"Exception: {ex}")
            raise
        finally:
            self._log_file.close()
            self._log_file = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rule")
    parser.add_argument("engine1")
    parser.add_argument("engine2")
    parser.add_argument("--log_prefix")
    args = parser.parse_args()
    rule = Rule.load(args.rule)
    log_prefix = args.log_prefix or f"data/auto_match/auto_match_{time.strftime('%Y%m%d%H%M%S')}_{os.getpid()}"
    engine_config_list = [EngineConfig.load(args.engine1), EngineConfig.load(args.engine2)]
    auto_match = AutoMatch(rule, engine_config_list)
    auto_match.run_matches(log_prefix)


if __name__ == "__main__":
    main()
