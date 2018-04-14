"""
置換表ファイルをマージする
"""

import os
import sys
import argparse
import subprocess
import sqlite3
import glob
import time
from collections import defaultdict
from logging import getLogger

logger = getLogger(__name__)

from typing import List, Dict, Tuple
from .. import util
from .make_book import Engine


def do_work(work_dir: str):
    config = util.yaml_load(os.path.join(work_dir, "config.yaml"))
    work_id = int(time.strftime('%Y%m%d%H%M%S'))

    engine = Engine(config, work_dir, work_id)
    for ttable_file in glob.glob(os.path.join(os.path.abspath(work_dir), "ttable", "*.bin")):
        if os.path.basename(ttable_file) == "merged.bin":
            continue
        print(ttable_file)
        engine.write_line(f"user load_tt {ttable_file}")
        print(engine.read_line())
    dst_path = os.path.join(os.path.abspath(work_dir), "ttable", "merged.bin")
    engine.write_line(f"user save_tt 1 {dst_path}")
    assert engine.read_line() == "ok"
    engine.quit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("work_dir")

    args = parser.parse_args()
    work_dir = args.work_dir
    do_work(work_dir)


if __name__ == '__main__':
    main()
