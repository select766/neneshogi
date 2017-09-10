"""
NNの学習
"""
import argparse

import numpy as np
import chainer

from .train_config import load_trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_dir")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()  # Make the GPU current

    trainer = load_trainer(args.config_dir, args.gpu)
    trainer.run()


if __name__ == '__main__':
    main()
