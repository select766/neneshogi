"""
設定ファイルからモデル学習のためのオブジェクトを生成
"""

import sys
import os
import shutil
import yaml
import warnings
import chainer
from chainer import training
from chainer.training import extensions

from . import config


def load_trainer(path, gpu: int = -1) -> training.Trainer:
    """
    :param path: 設定ファイルのあるディレクトリ
    :return:
    """
    with open(os.path.join(path, "solver.yaml")) as f:
        solver_yaml = yaml.load(f)

    id_from_dir = os.path.basename(path)
    id_from_solver_yaml = str(solver_yaml["id"])
    if id_from_dir != id_from_solver_yaml:
        warnings.warn(
            f"training id from directory name and solver.yaml mismatch: {id_from_dir} != {id_from_solver_yaml}")

    with open(os.path.join(path, solver_yaml["dataset_config"])) as f:
        dataset_yaml = yaml.load(f)

    with open(os.path.join(path, solver_yaml["model_config"])) as f:
        model_yaml = yaml.load(f)

    # pythonファイルをコピー
    code_dir = os.path.join(path, "code")
    os.makedirs(code_dir, exist_ok=True)
    code_src_dir = os.path.dirname(__file__)
    shutil.copy(os.path.join(code_src_dir, dataset_yaml["dataset_code"]), os.path.join(code_dir, "dataset.py"))
    shutil.copy(os.path.join(code_src_dir, model_yaml["model_code"]), os.path.join(code_dir, "model.py"))

    # pythonファイルをimport
    sys.path.insert(0, code_dir)
    import dataset
    import model

    # データセットローダーを生成
    dataset_loaders = {}
    for phase in ["train", "val"]:
        dataset_class = getattr(dataset, dataset_yaml[phase]["class"])
        dataset_loaders[phase] = dataset_class(**dataset_yaml[phase].get("kwargs", {}))

    # モデルを生成
    model_class = getattr(model, model_yaml["class"])
    model = model_class(**model_yaml.get("kwargs", {}))

    if gpu >= 0:
        model.to_gpu()

    train_iter = chainer.iterators.SerialIterator(
        dataset_loaders["train"], solver_yaml["batchsize"], repeat=True, shuffle=False
    )
    val_iter = chainer.iterators.SerialIterator(
        dataset_loaders["val"], solver_yaml["val_batchsize"], repeat=False, shuffle=False
    )

    optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)

    updater = training.StandardUpdater(train_iter, optimizer, device=gpu)

    weight_out_dir = os.path.join(path, "weight")
    trainer = training.Trainer(updater, (solver_yaml["epoch"], 'epoch'), weight_out_dir)

    val_interval = 1, 'epoch'
    log_interval = 100, 'iteration'

    trainer.extend(extensions.Evaluator(val_iter, model, device=gpu),
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

    return trainer