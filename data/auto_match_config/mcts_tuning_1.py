"""
MCTSプレイヤーの強さ比較
KPPTのやねうら王と対局させる。

テンプレートとなる設定ファイルを読み取り、ランダムに変えたい部分だけ変更してauto_match.pyを実行。
勝率50%以上なら、やねうら王の読みの深さを3手増加させて再実行。
"""

import sys
import os
import subprocess
import time
import argparse
import random
import copy
import shutil
import numpy as np
from neneshogi import util


def randlog10(a, b):
    """
    10^a ~ 10^bを均等に発生させる

    :param a:
    :param b:
    :return:
    """
    exp = random.uniform(a, b)
    return 10.0 ** exp


def generate_random_options(config_target):
    options = config_target["options"]
    options["c_puct"] = randlog10(-1, 1)
    options["softmax"] = randlog10(-1, 1)
    options["value_scale"] = randlog10(-1, 0)
    options["value_slope"] = randlog10(-1, 0.5)


def increment_base_strength(config_base):
    options = config_base["options"]
    options["DepthLimit"] += 3


def generate_run_dir(data_dir):
    path = os.path.join(data_dir, "run", f"{time.strftime('%Y%m%d%H%M%S')}_{os.getpid()}")
    os.mkdir(path)
    return path


def get_win_rate(output_path):
    auto_match_result = util.yaml_load(output_path)
    n_win = 0
    n_game = 0
    for match_result in auto_match_result.match_results:
        if match_result.winner == 1:
            n_win += 1
        n_game += 1
    return n_win / n_game


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("--n_configs", type=int, default=20)
    args = parser.parse_args()
    data_dir = args.data_dir
    assert os.path.exists(f"{data_dir}/config/rule.yaml")
    for i in range(args.n_configs):
        run_dir = generate_run_dir(args.data_dir)
        print(run_dir)
        rule_file_path = f"{run_dir}/rule.yaml"
        shutil.copy(f"{data_dir}/config/rule.yaml", rule_file_path)
        config_target_path = f"{run_dir}/engine_target.yaml"
        config_base = util.yaml_load(f"{data_dir}/config/engine_base.yaml")
        config_target = util.yaml_load(f"{data_dir}/config/engine_target.yaml")
        generate_random_options(config_target)
        print(config_target)
        util.yaml_dump(config_target, config_target_path)

        for strength in range(4):
            print("strength", strength)
            config_base_path = f"{run_dir}/engine_base_{strength}.yaml"
            util.yaml_dump(config_base, config_base_path)
            output_prefix = f"{run_dir}/result_{strength}"
            cmd = ["python", "-m", "neneshogi.auto_match",
                   rule_file_path, config_base_path, config_target_path,
                   "--log_prefix", output_prefix]
            subprocess.check_call(cmd)

            win_rate = get_win_rate(output_prefix + ".yaml")
            print("win_rate", win_rate)
            if win_rate >= 0.5:
                increment_base_strength(config_base)
            else:
                break


if __name__ == "__main__":
    main()
