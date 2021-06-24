import torch
import torch.nn as nn
import torch.nn.functional as F


class MyReLU(nn.Module):
    """nn.ReLU"""

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # y = (x > 0) ? x : 0
        return F.relu(x, inplace=self.inplace)
