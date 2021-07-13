import torch
import torch.nn as nn
import torch.nn.functional as F


class MyUpsamplingNearest2d(nn.Module):
    """nn.UpsamplingNearest2d"""

    def __init__(self,
                 size=None,
                 scale_factor=None):
        super().__init__()

        if (size is None) and (scale_factor is None):
            raise ValueError("Upsampling size and scale_factor cannot be both None.")

        self.size = size
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x:       (..., h, w)
        :return:
                        (..., h * scale_factor, w * scale_factor) or (..., size, size)
        """
        return F.interpolate(x, self.size, self.scale_factor,
                             mode="nearest", align_corners=None)


class MyUpsamplingBilinear2d(nn.Module):
    """nn.UpsamplingBilinear2d"""

    def __init__(self,
                 size=None,
                 scale_factor=None):
        super().__init__()

        if (size is None) and (scale_factor is None):
            raise ValueError("Upsampling size and scale_factor cannot be both None.")

        self.size = size
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x:       (..., h, w)
        :return:
                        (..., h * scale_factor, w * scale_factor) or (..., size, size)
        """
        return F.interpolate(x, self.size, self.scale_factor,
                             mode="bilinear", align_corners=True)
