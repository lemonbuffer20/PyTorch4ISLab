import torch
import torch.nn as nn
import torch.nn.functional as F


class MyLeakyReLU(nn.Module):
    """nn.LeakyReLU"""

    def __init__(self,
                 negative_slope: float = 0.01,
                 inplace: bool = False):
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # y = (x > 0) ? x : ax
        return F.leaky_relu(x, self.negative_slope, inplace=self.inplace)
