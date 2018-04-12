"""
モデル重みファイルの中身をfp16にする
"""

import os
import sys
import argparse
import numpy as np


def convert_fp16(src_path, dst_path):
    src_items = np.load(src_path)
    dst_items = {}
    for key in src_items.keys():
        dst_items[key] = src_items[key].astype(np.float16)

    with open(dst_path, "wb") as f:
        np.savez_compressed(f, **dst_items)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("dst")
    args = parser.parse_args()
    assert args.src != args.dst
    convert_fp16(args.src, args.dst)


if __name__ == "__main__":
    main()
