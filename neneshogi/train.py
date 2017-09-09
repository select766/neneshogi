"""
NNの学習
"""
import argparse

import numpy as np
import chainer

from chainer import training
from chainer.training import extensions

from .net import Model
from .kifu_dataset import PackedKifuDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_kifu")
    parser.add_argument("val_kifu")
    parser.add_argument("--train_size", type=int, required=True)
    parser.add_argument("--val_size", type=int, required=True)
    parser.add_argument("--train_offset", type=int, default=0)
    parser.add_argument("--val_offset", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batchsize", type=int, default=256)
    parser.add_argument("--val_batchsize", type=int, default=256)

    parser.add_argument('--epoch', '-E', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--resume', '-r', default='',
                        help='Initialize the trainer from given file')
    args = parser.parse_args()

    model = Model()
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()  # Make the GPU current
        model.to_gpu()

    # Load the datasets and mean file
    train = PackedKifuDataset(args.train_kifu, args.train_size, args.train_offset)
    val = PackedKifuDataset(args.val_kifu, args.val_size, args.val_offset)
    # These iterators load the images with subprocesses running in parallel to
    # the training/validation.
    train_iter = chainer.iterators.SerialIterator(
        train, args.batchsize, repeat=True, shuffle=False
    )
    val_iter = chainer.iterators.SerialIterator(
        val, args.val_batchsize, repeat=False, shuffle=False
    )
    # Set up an optimizer
    optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out)

    #val_interval = (10 if args.test else 100000), 'iteration'
    val_interval = 1, 'epoch'
    log_interval = 1000, 'iteration'

    trainer.extend(extensions.Evaluator(val_iter, model, device=args.gpu),
                   trigger=val_interval)
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=val_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'), trigger=val_interval)
    # Be careful to pass the interval directly to LogReport
    # (it determines when to emit log rather than when to read observations)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy', 'lr'
    ]), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

if __name__ == '__main__':
    main()
