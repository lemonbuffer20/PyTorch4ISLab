import torch
import torch.nn as nn
import torch.nn.functional as F


class MyDropout(nn.Module):

    def __init__(self,
                 drop_prob: float = 0.5,
                 inplace: bool = False):
        super().__init__()
        self.drop_prob = drop_prob
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or (self.drop_prob <= 0):  # shortcut
            return x

        return F.dropout(x, self.drop_prob, self.training, self.inplace)
