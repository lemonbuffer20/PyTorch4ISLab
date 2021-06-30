import torch
import torch.nn as nn


class MyGlobalAvgPool(nn.Module):

    def __init__(self, keepdim: bool = False):
        super().__init__()
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x:       (batch_size, num_channels, h, w)
        :return:        (batch_size, num_channels)
        """
        y = torch.mean(x, dim=[2, 3], keepdim=self.keepdim)
        return y
