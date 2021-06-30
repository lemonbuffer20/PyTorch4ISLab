import torch
import torch.nn as nn
import torch.nn.functional as F


class MyBatchNorm2d(nn.Module):

    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.9,
                 affine: bool = True) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.weight = self.bias = None

        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.running_mean: torch.Tensor
        self.running_var: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x:       (batch_size, num_features, h,w)
        :return:        (batch_size, num_features, h, w)
        """
        # The function will automatically handle BN.
        #
        # (1) during training... (self.training = True)
        #       y = weight * ((x - x_mean) / x_std) + bias
        #       running_mean' = running_mean * (1 - momentum) + x_mean * momentum
        #       running_var' = running_var * (1 - momentum) + x_var * momentum
        #       (x_mean = torch.mean(x, dim=[0, 2, 3])
        #       (x_var = torch.var(x, dim=[0, 2, 3])
        #
        # (2) during inference... (self.training = False)
        #       y = weight * ((x - running_mean) / running_std) + bias
        #       (running_std = 1/sqrt(running_var + eps))

        y = F.batch_norm(x,
                         self.running_mean,
                         self.running_var,
                         self.weight,
                         self.bias,
                         self.training,
                         self.momentum,
                         self.eps)
        return y
