"""
USI対応将棋エンジンとしてのエントリポイント

python -m neneshogi.neneshogi <engine name>
"""

import argparse
import logging
import os

import chainer

from . import config
from .usi import Usi
from .random_player import RandomPlayer
from .zero_search_player import ZeroSearchPlayer
from .one_search_player import OneSearchPlayer
from .simple_multi_serach_player import SimpleMultiSearchPlayer
from .narrow_search_player import NarrowSearchPlayer
from .prob_search_player import ProbSearchPlayer
from .alpha_beta_player import AlphaBetaPlayer
from .monte_carlo_softmax_v2_player import MonteCarloSoftmaxV2Player

engines = {"RandomPlayer": RandomPlayer,
           "ZeroSearchPlayer": ZeroSearchPlayer,
           "OneSearchPlayer": OneSearchPlayer,
           "SimpleMultiSearchPlayer": SimpleMultiSearchPlayer,
           "NarrowSearchPlayer": NarrowSearchPlayer,
           "ProbSearchPlayer": ProbSearchPlayer,
           "AlphaBetaPlayer": AlphaBetaPlayer,
           "MonteCarloSoftmaxV2Player": MonteCarloSoftmaxV2Player,
           }

profile_path = None


def main():
    logger = logging.getLogger("neneshogi")
    if profile_path is not None:
        logger.debug(f"Profile: {profile_path}")
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
    if os.environ.get("NENESHOGI_PROFILE", "0") == "1":
        import cProfile
        import time

        profile_path = os.path.join(config.PROFILE_DIR, f"cprofile_{time.strftime('%Y%m%d%H%M%S')}.bin")
        cProfile.run('main()', filename=profile_path)
    else:
        main()
