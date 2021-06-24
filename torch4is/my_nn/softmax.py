import torch
import torch.nn as nn


class MySoftmax(nn.Module):
    """nn.Softmax"""

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # y_i = exp(x_i) / sum_j(exp(x_j))
        return torch.softmax(x, dim=self.dim)
