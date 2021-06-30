import torch
import torch.nn as nn

from torch4is.my_model.resnet_block import BasicBlock
from torch4is.my_nn import MyConv2d, MyBatchNorm2d, MyReLU, MyLinear, MyGlobalAvgPool, MyDropout


class ResNet20(nn.Module):

    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 10) -> None:
        super().__init__()

        self.stem_conv = MyConv2d(in_channels, 16, 3, stride=1, padding=1, bias=False)
        self.stem_bn = MyBatchNorm2d(16)
        self.stem_relu = MyReLU(inplace=True)

        self.block10 = BasicBlock(16, 16, stride=1)
        self.block11 = BasicBlock(16, 16, stride=1)
        self.block12 = BasicBlock(16, 16, stride=1)

        self.block20 = BasicBlock(16, 32, stride=2)
        self.block21 = BasicBlock(32, 32, stride=1)
        self.block22 = BasicBlock(32, 32, stride=1)

        self.block30 = BasicBlock(32, 64, stride=2)
        self.block31 = BasicBlock(64, 64, stride=1)
        self.block32 = BasicBlock(64, 64, stride=1)

        self.global_pool = MyGlobalAvgPool()
        self.drop = MyDropout(0.1)
        self.fc = MyLinear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x:       (batch_size, in_channels, 32, 32)
        :return:        (batch_size, num_classes)
        """

        x = self.stem_conv(x)  # (3, 32, 32) -> (16, 32, 32)
        x = self.stem_bn(x)
        x = self.stem_relu(x)

        x = self.block10(x)  # (16, 32, 32) -> (16, 32, 32)
        x = self.block11(x)  # (16, 32, 32) -> (16, 32, 32)
        x = self.block12(x)  # (16, 32, 32) -> (16, 32, 32)

        x = self.block20(x)  # (16, 32, 32) -> (32, 16, 16)
        x = self.block21(x)  # (32, 16, 16) -> (32, 16, 16)
        x = self.block22(x)  # (32, 16, 16) -> (32, 16, 16)

        x = self.block30(x)  # (32, 16, 16) -> (64, 8, 8)
        x = self.block31(x)  # (64, 8, 8) -> (64, 8, 8)
        x = self.block32(x)  # (64, 8, 8) -> (64, 8, 8)

        x = self.global_pool(x)  # (64, 8, 8) -> (64,)
        x = self.drop(x)
        x = self.fc(x)  # (64,) -> (10,)
        return x
