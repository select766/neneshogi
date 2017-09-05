"""
ログ保存の初期化処理

ログはlog/neneshogi_<時刻>_<プロセスID>.logに出力される
ファイルとstderrで異なるログレベルに設定可能
"""

import os
import sys
import time
import logging

try:
    os.makedirs("log", exist_ok=True)
    log_stderr = logging.StreamHandler(sys.stderr)
    log_file = logging.FileHandler(f"log/neneshogi_{time.strftime('%Y%m%d%H%M%S')}_{os.getpid()}.log")
    log_stderr.setLevel(logging.WARN)
    log_file.setLevel(logging.DEBUG)
    rootLogger = logging.getLogger()
    log_stderr.setFormatter(logging.Formatter("%(asctime)s/%(name)s/%(levelname)s/%(message)s"))
    log_file.setFormatter(logging.Formatter("%(asctime)s/%(name)s/%(levelname)s/%(message)s"))
    rootLogger.addHandler(log_stderr)
    rootLogger.addHandler(log_file)
    rootLogger.setLevel(logging.DEBUG)
    logging.info("Logging started")
except Exception as ex:
    sys.stderr.write(f"Failed to initialize log! {ex}\n")
