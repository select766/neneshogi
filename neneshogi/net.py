# -*- coding: utf-8 -*-

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L


class ResBlock(chainer.Chain):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            conv1=L.Convolution2D(in_ch, out_ch, 3, pad=1, nobias=True),
            conv2=L.Convolution2D(out_ch, out_ch, 3, pad=1, nobias=True),
            conv3=L.Convolution2D(out_ch, out_ch, 3, pad=1, nobias=True),
            bn1=L.BatchNormalization(in_ch),
            bn2=L.BatchNormalization(out_ch),
            bn3=L.BatchNormalization(out_ch),
        )

    def __call__(self, x):
        y = self.conv1(F.relu(self.bn1(x)))
        y = F.dropout(self.conv2(F.relu(self.bn2(y))))
        y = self.conv3(F.relu(self.bn3(y)))

        return x + y


class Model(chainer.Chain):
    # モデルの設定
    def __init__(self, ch=16, depth=4, value_function=False):
        self.value_function = value_function
        out_ch = 1 if self.value_function else 27
        conv_first = L.Convolution2D(None, ch, 5, pad=2, nobias=True)
        bn_first = L.BatchNormalization(ch)
        res_blocks = chainer.ChainList(*[ResBlock(ch, ch) for i in range(depth)])
        if value_function:
            fc1 = L.Linear(None, 256)
            bn_fc1 = L.BatchNormalization(256)
            fc2 = L.Linear(None, out_ch)

            super().__init__(conv_first=conv_first,
                             bn_first=bn_first,
                             res_blocks=res_blocks,
                             fc1=fc1,
                             bn_fc1=bn_fc1,
                             fc2=fc2)
        else:
            conv_last = L.Convolution2D(None, out_ch, 1)
            super().__init__(conv_first=conv_first,
                             bn_first=bn_first,
                             res_blocks=res_blocks,
                             conv_last=conv_last)

        self.ch = ch
        self.depth = depth
        self.out_ch = out_ch

    # モデルを呼び出す
    def __call__(self, x):
        x = F.relu(self.bn_first(self.conv_first(x)))
        for i in range(len(self.res_blocks)):
            x = self.res_blocks[i](x)
        if self.value_function:
            x = F.dropout(F.relu(self.bn_fc1(self.fc1(x))))
            x = (F.reshape(self.fc2(x), (-1,)))
        else:
            x = self.conv_last(x)
            # x = F.reshape(x, (x.data.shape[0], -1))
        return x
