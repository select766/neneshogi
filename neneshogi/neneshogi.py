"""
USI対応将棋エンジンとしてのエントリポイント

python -m neneshogi.neneshogi <engine name>
"""

import argparse
import logging

import chainer

from .usi import Usi
from .random_player import RandomPlayer
from .zero_search_player import ZeroSearchPlayer
from .one_search_player import OneSearchPlayer
from .simple_multi_serach_player import SimpleMultiSearchPlayer
from .narrow_search_player import NarrowSearchPlayer

engines = {"RandomPlayer": RandomPlayer,
           "ZeroSearchPlayer": ZeroSearchPlayer,
           "OneSearchPlayer": OneSearchPlayer,
           "SimpleMultiSearchPlayer": SimpleMultiSearchPlayer,
           "NarrowSearchPlayer": NarrowSearchPlayer,
           }


def main():
    logger = logging.getLogger("neneshogi")
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("engine", choices=list(engines.keys()))
        parser.add_argument("--suffix", default="", help="Suffix to append to USI engine name")
        args = parser.parse_args()

        Usi.name_suffix = args.suffix

        engine = engines[args.engine]()
        chainer.config.use_cudnn = "never"  # TODO: bugfix
        logger.debug("Start USI")
        usi = Usi(engine)
        usi.run()
        logger.debug("Quit USI")
    except Exception as ex:
        logger.exception("Unhandled error")


if __name__ == "__main__":
    main()
