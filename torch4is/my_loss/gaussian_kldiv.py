import torch
import torch.nn as nn


class GaussianKLDiv(nn.Module):

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence loss to N(0, I) Gaussian
        :param mu:      (batch_size, latent_size)
        :param logvar:  (batch_size, latent_size)
        """
        loss = (torch.square(mu) + torch.exp(logvar) - 1 - logvar).div(2)

        if self.reduction == "none":
            return loss
        elif self.reduction == "sum":
            loss = torch.mean(loss, dim=-1).sum()
        elif self.reduction == "mean":
            loss = torch.mean(loss, dim=-1).mean()
        else:
            raise NotImplementedError
        return loss
