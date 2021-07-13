from typing import Tuple
import torch
import torch.nn as nn

from torch4is.my_nn import (MyConv2d, MyLeakyReLU, MyBatchNorm2d, MyLinear,
                            MyUpsamplingBilinear2d, MySigmoid)


class EncoderBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 negative_slope: float = 0.1,
                 *, stride: int = 2):
        super().__init__()

        if stride == 2:
            self.conv = MyConv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=False)
        else:  # stride == 1
            self.conv = MyConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = MyBatchNorm2d(out_channels)
        self.act = MyLeakyReLU(negative_slope=negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class DecoderBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 negative_slope: float = 0.1):
        super().__init__()

        self.up = MyUpsamplingBilinear2d(scale_factor=2)
        self.conv = MyConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = MyBatchNorm2d(out_channels)
        self.act = MyLeakyReLU(negative_slope=negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class MyVAE(nn.Module):

    def __init__(self,
                 in_channels: int = 3,
                 negative_slope: float = 0.1,
                 latent_size: int = 256):
        super().__init__()

        self.encoder = nn.Sequential(
            EncoderBlock(in_channels, 16, negative_slope, stride=1),  # (16, 16, 16)
            EncoderBlock(16, 32, negative_slope),  # (32, 16, 16)
            EncoderBlock(32, 64, negative_slope),  # (64, 8, 8)
            EncoderBlock(64, 128, negative_slope),  # (128, 4, 4)
        )

        self.latent_size = latent_size
        feature_size = 128 * 4 * 4  # 2048
        self.enc_proj = MyLinear(feature_size, latent_size * 2)
        self.mu = MyLinear(latent_size * 2, latent_size)
        self.logvar = MyLinear(latent_size * 2, latent_size)

        self.dec_proj = MyLinear(latent_size, feature_size)
        self.decoder = nn.Sequential(
            DecoderBlock(128, 64, negative_slope),  # (64, 8, 8)
            DecoderBlock(64, 32, negative_slope),  # (32, 16, 16)
            DecoderBlock(32, 16, negative_slope),  # (16, 32, 32)
            MyConv2d(16, in_channels, 3, 1, 1, bias=False),
            MySigmoid(),
        )

    @staticmethod
    def _generate_z(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(mu)  # random noise from N(0, 1)
        std = torch.exp(0.5 * logvar)
        z = mu + noise * std
        return z

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        x = self.encoder(x)  # (b, 128, 4, 4)

        x = x.view(batch_size, -1)  # (b, 128 * 4 * 4) = (b, 2048)
        x = self.enc_proj(x)
        mu = self.mu(x)  # (b, 256)
        logvar = self.logvar(x)  # (b, 256)
        return mu, logvar

    def generate(self, z: torch.Tensor) -> torch.Tensor:
        y = self.dec_proj(z)  # (b, 128 * 4 * 4)
        y = y.view(-1, 128, 4, 4)  # (b, 128, 4, 4)
        y = self.decoder(y)
        return y

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x:       (batch_size, 3, 32, 32)
        :return:
               x':      (batch_size, 3, 32, 32) -> reconstructed
               mu, logvar
        """
        mu, logvar = self.encode(x)
        z = self._generate_z(mu, logvar)  # (b, 256)
        y = self.generate(z)

        return y, mu, logvar
