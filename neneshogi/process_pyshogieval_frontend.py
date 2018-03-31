"""
yaneuraouプロセスから子プロセスとして呼ばれ、GPUクラスタプロセスを起動する。
system関数から呼ばれることを想定し、このプロセス自体はすぐに終了する。
通信用のMutexが生成されるまでは待機する。
"""
import argparse
import os
import sys
import time
import subprocess


def wait_mutex_created(name):
    """
    指定名称のmutexが生成されるまでブロックする。
    :param name:
    :return:
    """
    import win32con
    import win32event
    import pywintypes
    MUTEX_ALL_ACCESS = win32con.STANDARD_RIGHTS_REQUIRED | win32con.SYNCHRONIZE | win32con.MUTANT_QUERY_STATE
    while True:
        try:
            hWait = win32event.OpenMutex(MUTEX_ALL_ACCESS, False, name)
        except pywintypes.error:
            time.sleep(0.1)
        else:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue_prefix", default="neneshogi")
    args, thru_args = parser.parse_known_args()
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.Popen(["python", "-m", "neneshogi.process_pyshogieval_cluster"] + sys.argv[1:], cwd=project_dir)
    # 最も最後に初期化されるmutexを待つ
    mutex_name = args.queue_prefix + "_result_mutex"
    wait_mutex_created(mutex_name)
    # サブプロセスは終了しない


if __name__ == '__main__':
    main()
