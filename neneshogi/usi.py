"""
標準入出力でUSIプロトコルで通信する
現状、シングルスレッド
"""

import sys
from typing import Iterable, Dict, List

from .engine import Engine


class Usi:
    engine: Engine
    engine_options: Dict[str, str]

    def __init__(self, engine: Engine):
        self.engine = engine
        self.engine_options = {}

    def run(self):
        """
        コマンドループ
        :return: quitコマンドが来たらメソッドが終了する
        """
        for recv_line in sys.stdin:
            recv_line_nonl = recv_line.rstrip()
            tokens = recv_line_nonl.split(" ")  # type: List[str]
            if len(tokens) == 0:
                continue
            cmd = tokens[0]
            resp_lines = []
            if cmd == "usi":
                resp_lines.append(f"id name {self.engine.name}")
                resp_lines.append(f"id author {self.engine.author}")
                resp_lines.extend(f"option name {k} type {v}" for k, v in self.engine.get_options())
                resp_lines.append("usiok")
            elif cmd == "setoption":
                # setoption name USI_Ponder value true
                self.engine_options[tokens[2]] = tokens[4]
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
                        raise NotImplementedError(f"Unknown go option f{go_option_name}")
                bestmove = self.engine.go(**go_option_dict)
                resp_lines.append(f"bestmove {bestmove}")
            elif cmd == "gameover":
                # gameover win
                self.engine.gameover(tokens[1])
            elif cmd == "quit":
                self.engine.quit()
                break
            else:
                raise NotImplementedError(f"Unknown usi command f{cmd}")
            if len(resp_lines) > 0:
                self._put_lines(resp_lines)

    def _put_lines(self, lines: Iterable[str]) -> None:
        for line in lines:
            sys.stdout.write(line + "\n")
        sys.stdout.flush()
