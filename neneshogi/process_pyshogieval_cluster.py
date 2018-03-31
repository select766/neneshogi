"""
複数のGPU評価プロセスをまとめる
GPU番号の範囲および1GPUあたりいくつのプロセスを起動するかオプションで設定。
ホストプロセスが落ちたら全GPUプロセスを終了させる。
"""

from typing import Dict, Optional, List
import argparse
import subprocess
import psutil
import time

from logging import getLogger

logger = getLogger(__name__)


def exec_child(thru_args, gpu):
    args = ["python", "-m", "neneshogi.process_pyshogieval", "--gpu", str(gpu)] + thru_args
    proc = subprocess.Popen(args)
    logger.info(f"pid: {proc.pid}, args: {args}")
    return proc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_min", type=int, default=0)
    parser.add_argument("--gpu_max", type=int, default=0)
    parser.add_argument("--process_per_gpu", type=int, default=1)
    parser.add_argument("--host_pid", type=int)
    args, thru_args = parser.parse_known_args()

    logger.info("Starting cluster processes")
    children = []  # type: List[subprocess.Popen]
    for gpu in range(args.gpu_min, args.gpu_max + 1):
        for pg in range(args.process_per_gpu):
            children.append(exec_child(thru_args, gpu))

    logger.info("Waiting finish condition")
    if args.host_pid:
        while psutil.pid_exists(args.host_pid):
            time.sleep(1)
    else:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    logger.info("Killing cluster processes")
    for child in children:
        child.kill()

    for child in children:
        try:
            child.wait(timeout=1)
        except subprocess.TimeoutExpired:
            logger.warning(f"pid {child.pid} wait timeout")


if __name__ == '__main__':
    main()
