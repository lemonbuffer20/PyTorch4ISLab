import torch
import torch.nn as nn
import torch.nn.functional as F


class MyLinear(nn.Module):
    """nn.Linear"""

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_buffer('bias', None)  # self.bias = None

        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=2.0)
        if self.bias is not None:
            nn.init.uniform_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x:   (..., in_features)
        :return:    (..., out_features)
        """
        # (b, in_d) x (out_d, in_d)^T + (out_d,) = (b, out_d)
        y = F.linear(x, self.weight, self.bias)
        return y
