# -*- coding: utf-8 -*-
from typing import Tuple

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L


# 別ディレクトリにコピーして使われるので、相対import不可

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


class ModelRL(chainer.Chain):
    # モデルの設定
    def __init__(self, ch=16, depth=4, weight_move=1.0, weight_value=1.0):
        chains = {}
        chains["conv_first"] = L.Convolution2D(None, ch, 5, pad=2, nobias=True)
        chains["bn_first"] = L.BatchNormalization(ch)
        chains["res_blocks"] = chainer.ChainList(*[ResBlock(ch, ch) for i in range(depth)])
        chains["move_res_block"] = ResBlock(ch, ch)
        chains["move_conv"] = L.Convolution2D(None, 139, 3, pad=1)
        chains["value_res_block"] = ResBlock(ch, ch)
        chains["value_fc"] = L.Linear(None, 1)
        super().__init__(**chains)

        self.ch = ch
        self.depth = depth
        self.weight_move = weight_move
        self.weight_value = weight_value

    # モデルを呼び出す
    def forward(self, x: chainer.Variable) -> Tuple[chainer.Variable, chainer.Variable]:
        x = F.relu(self.bn_first(self.conv_first(x)))
        for i in range(len(self.res_blocks)):
            x = self.res_blocks[i](x)
        # move function
        m = self.move_res_block(x)
        m = self.move_conv(F.relu(m))
        m = F.reshape(m, (m.data.shape[0], -1))

        # value function
        v = self.value_res_block(x)
        v = self.value_fc(F.relu(v))
        v = F.flatten(v)

        return m, v

    def __call__(self, x, move_index, move_array, legal_move_array, eval_score, game_result):
        pred_move, pred_value = self.forward(x)
        # REINFORCE
        sce_move = F.softmax_cross_entropy(pred_move, move_index, reduce="no")  # log(選んだ指し手の確率)
        loss_move = F.mean(sce_move * game_result)
        loss_value = F.mean_squared_error(F.tanh(pred_value), game_result)

        loss_total = loss_move * self.weight_move + loss_value * self.weight_value
        accuracy = F.accuracy(pred_move, move_index)
        chainer.report({"loss": loss_total, "loss_move": loss_move, "loss_value": loss_value, "accuracy": accuracy},
                       self)
        return loss_total
