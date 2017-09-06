# -*- coding: utf-8 -*-

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L


class Model(chainer.Chain):
    # モデルの設定
    def __init__(self, ch=16, depth=4):
        convs = chainer.ChainList(
            *[L.Convolution2D(None, ch, 5, pad=2, nobias=True) for i in range(depth)])
        bns = chainer.ChainList(
            *[L.BatchNormalization(ch) for i in range(depth)])
        super_kwargs = {"convs": convs, "bns": bns, "conv": L.Convolution2D(None, 27, 5, pad=2)}
        super(Model, self).__init__(**super_kwargs)

        self.ch = ch
        self.depth = depth
        self.train = True

    # モデルを呼び出す
    def __call__(self, x):
        for i in range(len(self.convs)):
            x = self.convs[i](x)
            x = self.bns[i](x)
            x = F.relu(x)
        x = self.conv(x)
        #x = F.reshape(x, (x.data.shape[0], -1))
        return x
