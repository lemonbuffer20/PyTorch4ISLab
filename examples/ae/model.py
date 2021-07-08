from typing import Tuple
import torch
import torch.nn as nn

from torch4is.my_nn import (MyConv2d, MyReLU, MyLeakyReLU,
                            MyUpsamplingNearest2d, MyDropout, MySigmoid)


class MyConvAE(nn.Module):

    def __init__(self,
                 in_channels: int = 3,
                 negative_slope: float = 0.02,
                 drop_prob: float = 0.1):
        super().__init__()

        if negative_slope <= 0.0:
            self.act = MyReLU(inplace=True)
        else:
            self.act = MyLeakyReLU(negative_slope=negative_slope, inplace=True)
        self.drop = MyDropout(drop_prob=drop_prob)

        self.conv10 = MyConv2d(in_channels, 16, 4, 2, 1, bias=True)
        self.conv20 = MyConv2d(16, 32, 4, 2, 1, bias=True)
        self.conv30 = MyConv2d(32, 64, 4, 2, 1, bias=True)

        self.up = MyUpsamplingNearest2d(scale_factor=2)

        self.conv40 = MyConv2d(64, 64, 1, 1, 0, bias=True)
        self.conv41 = MyConv2d(64, 64, 1, 1, 0, bias=True)

        self.conv50 = MyConv2d(64, 32, 3, 1, 1, bias=True)
        self.conv60 = MyConv2d(32, 16, 3, 1, 1, bias=True)
        self.conv70 = MyConv2d(16, in_channels, 3, 1, 1, bias=False)

        self.sigmoid = MySigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x:       (batch_size, 3, 32, 32)
        :return:
               x':      (batch_size, 3, 32, 32)
               feat:    (batch_size, 64, 4, 4) -> flatten will give 64x4x4 = 1024-dim feature. (not much compressed)
        """
        x = self.conv10(x)  # (3, h, w) -> (16, h/2, w/2)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv20(x)  # (16, h/2, w/2) -> (32, h/4, w/4)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv30(x)  # (32, h/4, w/4) -> (64, h/8, w/8)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv40(x)  # (64, h/8, w/8) -> (64, h/8, w/8)

        feat = x

        x = self.conv41(x)  # (64, h/8, w/8) -> (64, h/8, w/8)
        x = self.act(x)
        x = self.drop(x)
        x = self.up(x)
        x = self.conv50(x)  # (64, h/4, w/4) -> (32, h/4, w/4)
        x = self.act(x)
        x = self.drop(x)
        x = self.up(x)
        x = self.conv60(x)  # (32, h/2, w/2) -> (16, h/2, w/2)
        x = self.act(x)
        x = self.drop(x)
        x = self.up(x)
        x = self.conv70(x)  # (16, h, w) -> (3, h, w)

        x = self.sigmoid(x)
        return x, feat
