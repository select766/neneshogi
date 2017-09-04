"""
盤面クラスのテストデータ作成ツール

棋譜を読み込み、各局面の合法手リストを生成する。
生成には、やねうら王に改造を施したプログラムを使用。
YAML形式で作成するが、ロードが遅いのでpickleに変換したものを保存

入力棋譜ファイルの形式
startpos moves 7g7f 8c8d ...
詰みまで含まれていることが望ましい。
"""

import sys
import os
import argparse
import pickle
import subprocess
from typing import List, Optional, Tuple

import yaml

from . import config


class YaneuraOuHandler:
    def __init__(self, exe_path):
        self.process = subprocess.Popen([exe_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self._send_init()

    def _send_command(self, command: str):
        self.process.stdin.write((command + "\n").encode("ascii"))
        self.process.stdin.flush()

    def _send_and_read(self, command: str, end_of_response: Optional[str]) -> List[str]:
        """
        コマンドを1行送り、特定内容の行が現れるまで出力を受け取る。
        :param command:
        :param end_of_response: Noneの場合は何も読み取らず戻る。
        :return:
        """
        self._send_command(command)
        if end_of_response is None:
            return []
        lines = []
        while True:
            line = self.process.stdout.readline().decode("ascii").rstrip()
            if line == end_of_response:
                break
            lines.append(line)
        return lines

    def _send_init(self):
        self._send_and_read("usi", "usiok")
        self._send_and_read("isready", "readyok")

    def close(self):
        """
        プロセスを終了する
        :return:
        """
        self._send_command("quit")
        self.process.wait()

    def generate_case(self, position_command: str) -> List[str]:
        """
        1つのposition commandに対する合法手データ生成
        :param position_command:
        :return: やねうら王から得られた各行をそのまま返す
        """
        self._send_command(position_command)
        lines = self._send_and_read("user", "---")
        return lines


def generate_case_game(kifu: str, yane_handler: YaneuraOuHandler, serial: int) -> Tuple[str, int]:
    """
    1回の対局に関してテストケースを生成する
    :param kifu: "startpos moves 7g7f ..."
    :param yane_handler:
    :return: YAMLおよび新しいシリアル番号
    """
    pos_parts = kifu.rstrip().split(" ")
    cases = ""
    for i in range(2, len(pos_parts) + 1):
        position_command = "position " + " ".join(pos_parts[:i])
        yane_lines = yane_handler.generate_case(position_command)
        # シリアル番号と入力コマンドを付加
        yane_lines.append(f"serial: {serial}")
        serial += 1
        yane_lines.append(f"position_command: \"{position_command}\"")
        # インデントをつけて出力
        cases += "-\n"
        for line in yane_lines:
            cases += f"  {line}\n"
    return cases, serial


def generate_position_testcase():
    parser = argparse.ArgumentParser()
    parser.add_argument("kifu")
    parser.add_argument("--dst", default=config.MOVEGEN_TESTCAST_PATH)
    args = parser.parse_args()

    yane_handler = YaneuraOuHandler(config.YANEURAOU_MOVEGEN_EXE)
    serial = 0
    with open(args.kifu) as kifu_f:
        with open(args.dst + ".yaml", "w") as dst_f:
            for kifu_line in kifu_f:
                game_yaml, serial = generate_case_game(kifu_line, yane_handler, serial)
                dst_f.write(game_yaml)
    yane_handler.close()

    # YAMLのロードが非常に遅いので、pickleに変換しておく
    with open(args.dst + ".yaml") as dst_f:
        with open(args.dst, "wb") as pickle_f:
            yaml_data = yaml.load(dst_f)
            pickle.dump(yaml_data, pickle_f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    generate_position_testcase()
