"""
USIエンジンを走らせ、悪手の出現統計をとる。
"""
import argparse
import pickle
import time
import os
import sys

import shutil
import yaml
import subprocess

from .. import config
from .move_comparator import EvalOutput


def calculate_statistics(move_compare_data, stat_config):
    count_match = 0
    count_except = 0
    count_not_bad = 0
    count_bad = 0
    for record in move_compare_data:
        root_t_eval = record["root_t_eval"]  # type: EvalOutput
        root_s_eval = record["root_s_eval"]  # type: EvalOutput
        t_after_t_eval = record["t_after_t_eval"]  # type: EvalOutput
        t_after_s_eval = record["t_after_s_eval"]  # type: EvalOutput
        s_after_t_eval = record["s_after_t_eval"]  # type: EvalOutput
        s_after_s_eval = record["s_after_s_eval"]  # type: EvalOutput

        if root_t_eval.bestmove == root_s_eval.bestmove:
            # 指し手一致
            count_match += 1
        else:
            # 指し手不一致
            if t_after_t_eval.score_type != "cp" or s_after_t_eval.score_type != "cp":
                # 通常の評価値が出ていないものは判定できない
                count_except += 1
            else:
                # 1手先同士の評価値の差を計算
                # 生徒の手が悪手なら、手を指した後の相手側の手番における評価値が大きくなる。
                score_diff = s_after_t_eval.score - t_after_t_eval.score
                if score_diff >= stat_config["bad_move_threshold"]:
                    count_bad += 1
                else:
                    count_not_bad += 1
    return {"match": count_match, "except": count_except, "bad": count_bad, "not_bad": count_not_bad}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("teacher")
    parser.add_argument("student")
    parser.add_argument("--dst")
    args = parser.parse_args()

    if args.dst:
        dst_dir = args.dst
    else:
        dst_id = time.strftime("%Y%m%d%H%M%S")
        dst_dir = os.path.join(config.BAD_MOVE_STATICS_OUTPUT_DIR, dst_id)
        os.makedirs(dst_dir, exist_ok=False)

    sys.stderr.write(f"Output: {dst_dir}\n")
    with open(args.config) as f:
        stat_config = yaml.load(f)

    shutil.copy2(args.config, os.path.join(dst_dir, "config.yaml"))
    shutil.copy2(args.teacher, os.path.join(dst_dir, "teacher.txt"))
    shutil.copy2(args.student, os.path.join(dst_dir, "student.txt"))

    move_compare_file = os.path.join(dst_dir, "move_compare.pickle")

    if not os.path.exists(move_compare_file):
        subprocess.check_call(["python", "-m", "neneshogi.tools.move_comparator",
                               stat_config["kifu"], move_compare_file,
                               "--teacher", args.teacher,
                               "--student", args.student,
                               "--skipfirst", str(stat_config.get("skipfirst", 0)),
                               "--skiplast", str(stat_config.get("skiplast", 0)),
                               "--games", str(stat_config.get("games", -1))])
    else:
        sys.stderr.write("Using existing compare file\n")

    with open(move_compare_file, "rb") as f:
        move_compare_data = pickle.load(f)
    stats = calculate_statistics(move_compare_data, stat_config)
    info = {"stats": stats}
    try:
        # gitのcommit番号を取得する
        info["commit"] = subprocess.getoutput("git show -s --format=%H")
    except:
        pass

    stat_save_file = os.path.join(dst_dir, "stat.yaml")
    with open(stat_save_file, "w") as f:
        yaml.dump(info, f, default_flow_style=False)

    count_total = stats["match"] + stats["except"] + stats["bad"] + stats["not_bad"]
    sys.stdout.write(f"match: {int(stats['match']/count_total*100)}%, " +
                     f"bad: {int(stats['bad']/count_total*100)}%, " +
                     f"not_bad: {int(stats['not_bad']/count_total*100)}%, " +
                     f"except: {int(stats['except']/count_total*100)}%\n")


if __name__ == "__main__":
    main()
