"""
バイナリ棋譜データのシャッフル
"""

import sys
import argparse
import numpy as np


def read_shuffle_write(srcf, dstf, record_size, count):
    data = np.fromfile(srcf, dtype=np.uint8, count=record_size * count)
    data = data.reshape((count, record_size))
    shuffled_data = data[np.random.permutation(count)]
    shuffled_data.tofile(dstf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--record_size", type=int, default=36)
    parser.add_argument("--count", type=int, default=1024 * 1024 * 10)
    parser.add_argument("--buffer_size", type=int, default=1024 * 1024 * 1024)
    args = parser.parse_args()

    assert args.src != args.dst
    buffer_count = args.buffer_size // args.record_size
    remain_count = args.count

    with open(args.src, "rb") as srcf:
        with open(args.dst, "wb") as dstf:
            while remain_count > 0:
                count = min(buffer_count, remain_count)
                read_shuffle_write(srcf, dstf, args.record_size, count)
                remain_count -= count


if __name__ == '__main__':
    main()
