"""
標準入出力でUSIプロトコルで通信する
現状、シングルスレッド
"""

import sys
import queue
import threading
from typing import Iterable, Dict, List

from .engine import Engine

from logging import getLogger

logger = getLogger(__name__)

from .usi_info_writer import UsiInfoWriter


class Usi:
    engine: Engine
    engine_options: Dict[str, str]
    # 将棋所で同一エンジン別設定を行うときに、区別がつくようにエンジン名にsuffixをつける機能
    name_suffix = ""

    def __init__(self, engine: Engine):
        self.engine = engine
        self.engine_options = {}
        self.stdin_thread = None
        self.stdin_queue = None

    def run(self):
        """
        コマンドループ
        :return: quitコマンドが来たらメソッドが終了する
        """
        info_writer = UsiInfoWriter(self._put_lines)
        self.stdin_queue = queue.Queue()
        self.stdin_thread = threading.Thread(target=stdin_thread, args=(self.stdin_queue,))
        self.stdin_thread.start()
        while True:
            recv_line = self.stdin_queue.get()
            recv_line_nonl = recv_line.rstrip()
            logger.info(f"USI< {recv_line_nonl}")
            tokens = recv_line_nonl.split(" ")  # type: List[str]
            if len(tokens) == 0:
                continue
            cmd = tokens[0]
            resp_lines = []
            if cmd == "usi":
                resp_lines.append(f"id name {self.engine.name+Usi.name_suffix}")
                resp_lines.append(f"id author {self.engine.author}")
                resp_lines.extend(f"option name {k} type {v}" for k, v in self.engine.get_options().items())
                resp_lines.append("usiok")
            elif cmd == "setoption":
                # setoption name USI_Ponder value true
                self.engine_options[tokens[2]] = tokens[4] if len(tokens) >= 5 else ""
            elif cmd == "isready":
                self.engine.isready(self.engine_options)
                resp_lines.append("readyok")
            elif cmd == "usinewgame":
                self.engine.usinewgame()
            elif cmd == "position":
                self.engine.position(recv_line_nonl)
            elif cmd == "go":
                tokens.pop(0)
                go_option_dict = {}
                while len(tokens) > 0:
                    go_option_name = tokens.pop(0)
                    if go_option_name in ["btime", "wtime", "byoyomi", "binc", "winc"]:
                        go_option_dict[go_option_name] = int(tokens.pop(0))
                    else:
                        raise NotImplementedError(f"Unknown go option {go_option_name}")
                bestmove = self.engine.go(info_writer, **go_option_dict)
                resp_lines.append(f"bestmove {bestmove}")
            elif cmd == "gameover":
                # gameover win
                self.engine.gameover(tokens[1])
            elif cmd == "quit":
                self.engine.quit()
                break
            else:
                raise NotImplementedError(f"Unknown usi command {cmd}")
            if len(resp_lines) > 0:
                self._put_lines(resp_lines)
            if cmd == "go":
                # msvcrt.kbhit(): stdinではなくキーボード入力しか見られなくてダメ
                self.engine.ponder(bestmove, lambda: not self.stdin_queue.empty())

    def _put_lines(self, lines: Iterable[str]) -> None:
        for line in lines:
            logger.info(f"USI> {line}")
            sys.stdout.write(line + "\n")
        sys.stdout.flush()


def stdin_thread(q: queue.Queue):
    for recv_line in sys.stdin:
        q.put(recv_line)
        if recv_line.startswith("quit"):
            break
