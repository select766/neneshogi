"""
盤面表現・合法手生成のテスト

Test case:
  side_to_move: 1
  piece_on: [18,0,17,0,0,0,1,0,2,19,21,17,0,0,0,1,0,3,20,0,17,0,0,0,1,0,4,23,0,17,0,0,0,1,0,7,24,0,17,0,0,0,1,0,8,23,0,17,0,0,0,1,6,7,20,0,17,0,0,0,1,0,4,19,22,17,0,0,0,1,5,3,18,0,17,0,0,0,1,0,2,]
  hand_of: [[0,0,0,0,0,0,0,],[0,0,0,0,0,0,0,],]
  in_check: false
  sfen: "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B1R5/LNSGKGSNL w - 2"
  legal_moves: ["1c1d","2c2d","3c3d","4c4d","5c5d","6c6d","7c7d","8c8d","9c9d","1a1b","9a9b","3a3b","3a4b","7a6b","7a7b","8b3b","8b4b","8b5b","8b6b","8b7b","8b9b","4a3b","4a4b","4a5b","5a4b","5a5b","5a6b","6a5b","6a6b","6a7b",]
  serial: 1
  position_command: "position startpos moves 2h6h"
"""

import unittest
from typing import List

import pickle

from neneshogi import config
from neneshogi.position import Position

class TestPosition(unittest.TestCase):
    pos: Position
    dataset: List

    def setUp(self):
        self.pos = Position()
        with open(config.MOVEGEN_TESTCAST_PATH, "rb") as f:
            self.dataset = pickle.load(f)

    def test_position_command(self):
        """
        Positionコマンドで設定した盤面と、SFENを比較する
        :return:
        """
        for case in self.dataset:
            self.pos.set_usi_position(case["position_command"])
            self.assertEqual(self.pos.get_sfen(), case["sfen"], f"Case {case['serial']}")
