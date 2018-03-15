"""
C++実装の評価関数を試すためのMCTS player
"""

from typing import Dict, Optional, List, Tuple
import json
import subprocess
import os

from logging import getLogger

logger = getLogger(__name__)

import numpy as np
import chainer
import chainer.functions as F

from .position import Position, Color, Square, Piece, Move, PositionHelper
from yaneuraou import DNNConverter
from .engine import Engine
from .usi_info_writer import UsiInfoWriter
from .train_config import load_model
from . import util
from .mcts_naive_player import MCTSNaivePlayer, TreeNode


class MCTSNaivePlayerExe(MCTSNaivePlayer):
    usi_process: subprocess.Popen

    def __init__(self):
        super().__init__()
        usi_path = r"D:\dev\shogi\YaneuraOu_mcts\build\user\YaneuraOu-user.exe"

        usi_process = subprocess.Popen([usi_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       cwd=os.path.dirname(usi_path), encoding="utf-8")
        usi_process.stdin.write("usi\n")
        usi_process.stdin.flush()
        while True:
            line = usi_process.stdout.readline().rstrip()
            if line == "usiok":
                break
        usi_process.stdin.write("isready\n")
        usi_process.stdin.flush()
        while True:
            line = usi_process.stdout.readline().rstrip()
            if line == "readyok":
                break
        self.usi_process = usi_process

    @property
    def name(self):
        return "NeneShogi MCTSNaiveExe"

    def _eval_current_pos(self, parent: "TreeNode", parent_edge_index: int) -> TreeNode:
        move_list = self.pos.generate_move_list()
        if len(move_list) == 0:
            # mated
            return TreeNode(self.tree_config, parent, parent_edge_index, move_list, -1.0, None)
        sfen = self.pos.sfen()
        self.usi_process.stdin.write(f"position sfen {sfen}\n")
        self.usi_process.stdin.write(f"user eval\n")
        self.usi_process.stdin.flush()
        eval_result = json.loads(self.usi_process.stdout.readline().rstrip())
        # 空文字列キー=勝敗、usiの指し手をキーとした数値:指し手確率
        value_p = []
        for move in move_list:
            value_p.append(eval_result[str(move)])
        return TreeNode(self.tree_config, parent, parent_edge_index, move_list,
                        eval_result[""], np.array(value_p, dtype=np.float32))

    def quit(self):
        super().quit()
        self.usi_process.stdin.write("quit\n")
        self.usi_process.stdin.flush()
