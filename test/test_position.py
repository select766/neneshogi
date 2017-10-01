"""
盤面表現・合法手生成のテスト

合法手生成のテストのみ:
python -m unittest test.test_position.TestPosition.test_move_generation

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

import numpy as np
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
        Positionコマンドで設定した盤面と、駒配置およびSFENを比較する
        :return:
        """
        self.pos.set_usi_position("position startpos")
        self.assertEqual(self.pos.get_sfen(), "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
        for case in self.dataset:
            self.pos.set_usi_position(case["position_command"])
            self.assertEqual(self.pos.side_to_move, case["side_to_move"], f"Case {case['serial']}")
            self.assertTrue(np.all(self.pos.board == np.array(case["piece_on"], dtype=np.uint8)),
                            f"Case {case['serial']}")
            self.assertTrue(np.all(self.pos.hand == np.array(case["hand_of"], dtype=np.uint8)),
                            f"Case {case['serial']}")
            self.assertTrue(self.pos.in_check() == case["in_check"],
                            f"Case {case['serial']}")
            self.assertEqual(self.pos.get_sfen(), case["sfen"], f"Case {case['serial']}")

    def test_move_generation(self):
        """
        合法手生成のテスト
        :return:
        """
        for case in self.dataset:
            self.pos.set_usi_position(case["position_command"])
            legal_moves = self.pos.generate_move_list()
            legal_moves_str = [m.to_usi_string() for m in legal_moves]
            self.assertSetEqual(set(legal_moves_str), set(case["legal_moves"]), f"Case {case['serial']}")

    def test_position_command_sfen(self):
        """
        Positionコマンドの初期局面が"startpos"ではなくsfenの場合の局面再生テスト
        :return:
        """
        self.pos.set_usi_position(
            "position sfen lnsg2snl/3kg1rb1/pppppp1pp/9/9/2PP5/PP2PP+pPP/1B1R2SK1/LNSG1G1NL b p 15")
        self.assertEqual(self.pos.get_sfen(), "lnsg2snl/3kg1rb1/pppppp1pp/9/9/2PP5/PP2PP+pPP/1B1R2SK1/LNSG1G1NL b p 15")
        self.pos.set_usi_position(
            "position sfen lnsg2snl/3kg1rb1/pppppp1pp/9/9/2PP5/PP2PP+pPP/1B1R2SK1/LNSG1G1NL b p 15 moves 3h3g")
        self.assertEqual(self.pos.get_sfen(), "lnsg2snl/3kg1rb1/pppppp1pp/9/9/2PP5/PP2PPSPP/1B1R3K1/LNSG1G1NL w Pp 16")
        self.pos.set_usi_position(
            "position sfen lnsg2snl/3kg1rb1/pppppp1pp/9/9/2PP5/PP2PP+pPP/1B1R2SK1/LNSG1G1NL b p 15 moves 3h3g P*3f")
        self.assertEqual(self.pos.get_sfen(), "lnsg2snl/3kg1rb1/pppppp1pp/9/9/2PP2p2/PP2PPSPP/1B1R3K1/LNSG1G1NL b P 17")

    def test_sfen_set_get(self):
        """
        SFENでの局面設定が正しいかテスト
        :return:
        """
        for case in self.dataset:
            self.pos.set_sfen(case["sfen"])
            self.assertEqual(self.pos.side_to_move, case["side_to_move"], f"Case {case['serial']}")
            self.assertTrue(np.all(self.pos.board == np.array(case["piece_on"], dtype=np.uint8)),
                            f"Case {case['serial']}")
            self.assertTrue(np.all(self.pos.hand == np.array(case["hand_of"], dtype=np.uint8)),
                            f"Case {case['serial']}")
            self.assertTrue(self.pos.in_check() == case["in_check"],
                            f"Case {case['serial']}")
            self.assertEqual(self.pos.get_sfen(), case["sfen"], f"Case {case['serial']}")
