"""
設定ファイルからモデル学習のためのオブジェクトを生成
"""
import argparse
import sys
import os
import shutil

import time
import yaml
import warnings
import chainer
from chainer import training
from chainer.training import extensions

from . import config
from . import util


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

    with open(os.path.join(path, "dataset.yaml")) as f:
        dataset_yaml = yaml.load(f)

    with open(os.path.join(path, "model.yaml")) as f:
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
    initmodel_path = solver_yaml.get("initmodel", None)
    if initmodel_path:
        chainer.serializers.load_npz(initmodel_path, model)

    if gpu >= 0:
        model.to_gpu()

    train_iter = chainer.iterators.SerialIterator(
        dataset_loaders["train"], solver_yaml["batchsize"], repeat=True, shuffle=False
    )
    val_iter = chainer.iterators.SerialIterator(
        dataset_loaders["val"], solver_yaml["val_batchsize"], repeat=False, shuffle=False
    )

    optimizer_class = {"MomentumSGD": chainer.optimizers.MomentumSGD}[solver_yaml["optimizer"]]
    optimizer = optimizer_class(**solver_yaml.get("optimizer_kwargs", {}))
    optimizer.setup(model)

    updater = training.StandardUpdater(train_iter, optimizer, device=gpu)

    weight_out_dir = os.path.join(path, "weight")
    trainer = training.Trainer(updater, (solver_yaml["epoch"], 'epoch'), weight_out_dir)

    if "val_interval_iteration" in solver_yaml:
        val_interval = solver_yaml["val_interval_iteration"], "iteration"
    else:
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
        'main/loss_move', 'validation/main/loss_move',
        'main/loss_value', 'validation/main/loss_value',
        'main/accuracy', 'validation/main/accuracy', 'lr'
    ]), trigger=log_interval)

    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'iteration', val_interval, file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(['main/loss_move', 'validation/main/loss_move'],
                                  'iteration', val_interval, file_name='loss_move.png'))
        trainer.extend(
            extensions.PlotReport(['main/loss_value', 'validation/main/loss_value'],
                                  'iteration', val_interval, file_name='loss_value.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'iteration', val_interval, file_name='accuracy.png'))

    trainer.extend(extensions.ProgressBar(update_interval=10))

    return trainer


def load_model(model_path: str) -> chainer.Chain:
    """
    学習済みモデルをロードする。
    :param model_path: ウェイトファイルのパス
    :return:
    """
    model_path = util.strip_path(model_path)
    model_dir = os.path.dirname(os.path.dirname(model_path))
    code_dir = os.path.join(model_dir, "code")

    with open(os.path.join(model_dir, "model.yaml")) as f:
        model_yaml = yaml.load(f)

    # pythonファイルをimport
    sys.path.insert(0, code_dir)
    import model

    # モデルを生成
    model_class = getattr(model, model_yaml["class"])
    model = model_class(**model_yaml.get("kwargs", {}))

    chainer.serializers.load_npz(model_path, model)

    return model


def command_clone():
    """
    設定ファイルをコピーする
    yamlファイルのコピーと、新規idの設定

    python -m neneshogi.train_config clone 201712345678
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("src_id")
    args = parser.parse_args()
    dst_id = time.strftime("%Y%m%d%H%M%S")
    src_dir = os.path.join(config.MODEL_OUTPUT_DIR, args.src_id)
    dst_dir = os.path.join(config.MODEL_OUTPUT_DIR, dst_id)
    os.makedirs(dst_dir, exist_ok=False)
    for name in ["dataset.yaml", "model.yaml", "solver.yaml"]:
        shutil.copy(os.path.join(src_dir, name), os.path.join(dst_dir, name))
    with open(os.path.join(dst_dir, "solver.yaml"), "r") as f:
        solver_yaml = yaml.load(f)
    solver_yaml["id"] = dst_id
    with open(os.path.join(dst_dir, "solver.yaml"), "w") as f:
        yaml.dump(solver_yaml, f, default_flow_style=False)


def main():
    command = sys.argv[1]
    sys.argv.pop(1)
    if command == "clone":
        command_clone()
    else:
        raise NotImplementedError(f"Unknown command {command}")


if __name__ == '__main__':
    main()
