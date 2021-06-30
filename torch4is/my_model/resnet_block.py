import torch
import torch.nn as nn

from torch4is.my_nn import MyConv2d, MyBatchNorm2d, MyReLU


class BasicBlock(nn.Module):

    def __init__(self,
                 in_ch: int,
                 res_ch: int,
                 stride: int = 1) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.res_ch = res_ch

        if stride not in (1, 2):
            raise ValueError(f"[ERROR:MODEL] ResNet should have either stride 1 or 2.")
        self.stride = stride

        self.conv1 = MyConv2d(in_ch, res_ch, 3, stride, padding=1, bias=False)
        self.bn1 = MyBatchNorm2d(res_ch)
        self.relu1 = MyReLU(inplace=True)

        self.conv2 = MyConv2d(res_ch, res_ch, 3, 1, padding=1, bias=False)
        self.bn2 = MyBatchNorm2d(res_ch)

        self.has_shortcut = ((stride != 1) or (in_ch != res_ch))
        if self.has_shortcut:
            self.shortcut_conv = MyConv2d(in_ch, res_ch, 1, stride, padding=0, bias=False)
            self.shortcut_bn = MyBatchNorm2d(res_ch)
        else:
            self.shortcut_conv = self.shortcut_bn = None

        self.relu_out = MyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x:       (batch_size, in_ch, h, w)
        :return:        (batch_size, res_ch, h (or h/2), w (or w/2))
        """
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.has_shortcut:
            identity = self.shortcut_conv(identity)
            identity = self.shortcut_bn(identity)

        x = x + identity
        x = self.relu_out(x)
        return x
