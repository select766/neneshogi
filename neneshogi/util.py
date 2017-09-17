import string
import functools

import chainer


def strip_path(path: str) -> str:
    """
    ファイルパスの両端から不要な文字を除去する。
    Windowsで便利なように、両端から",'を除去する。
    :param path:
    :return:
    """
    # 「パスのコピー」でダブルクオーテーションが入るのでそれをそのまま利用したい
    return path.strip(string.whitespace + "'\"")


def release_gpu_memory_pool(func):
    """
    デコレータとして使用し、関数の実行後にGPUメモリプールを解放する。
    :param func:
    :return:
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        if self.gpu >= 0 and hasattr(chainer.cuda, "memory_pool"):
            # 2つのプログラムを同時に走らせて対戦させたときに、memory poolがGPUメモリを持ったままだと相手側にメモリ不足が生じる
            # TODO: ponderを実装したらこれでは解決しない
            chainer.cuda.memory_pool.free_all_blocks()
        return result

    return wrapper
