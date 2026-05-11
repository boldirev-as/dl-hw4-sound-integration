import math

import torch
from torch import nn


def get_padding(kernel_size, dilation=1):
    return (kernel_size - 1) * dilation // 2


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return x + torch.sin(self.alpha * x).pow(2) / (self.alpha + 1e-9)


class ResidualUnit(nn.Module):
    def __init__(self, channels, dilation, kernel_size=7):
        super().__init__()
        hidden_channels = channels // 2
        self.block = nn.Sequential(
            Snake1d(channels),
            nn.Conv1d(
                channels,
                hidden_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=get_padding(kernel_size, dilation),
            ),
            Snake1d(hidden_channels),
            nn.Conv1d(hidden_channels, channels, kernel_size=1),
        )

    def forward(self, x):
        return x + self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, channels, stride, dilations, kernel_size):
        super().__init__()
        self.residual_layers = nn.Sequential(
            ResidualUnit(channels // 2, dilations[0], kernel_size),
            ResidualUnit(channels // 2, dilations[1], kernel_size),
            ResidualUnit(channels // 2, dilations[2], kernel_size),
            Snake1d(channels // 2),
            nn.Conv1d(
                channels // 2,
                channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.residual_layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, channels, stride, dilations, kernel_size):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(channels),
            nn.ConvTranspose1d(
                channels,
                channels // 2,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=(2 * math.ceil(stride / 2)) - stride,
            ),
            ResidualUnit(channels // 2, dilations[0], kernel_size),
            ResidualUnit(channels // 2, dilations[1], kernel_size),
            ResidualUnit(channels // 2, dilations[2], kernel_size),
        )

    def forward(self, x):
        return self.block(x)


class SEANetEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        base_channels,
        latent_dim,
        strides,
        dilations,
        residual_kernel_size,
    ):
        super().__init__()
        layers = [nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3)]
        current_channels = base_channels
        for stride in strides:
            next_channels = current_channels * 2
            layers.append(EncoderBlock(next_channels, stride, dilations, residual_kernel_size))
            current_channels = next_channels
        layers.extend(
            [
                Snake1d(current_channels),
                nn.Conv1d(current_channels, latent_dim, kernel_size=3, padding=1),
            ]
        )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class SEANetDecoder(nn.Module):
    def __init__(
        self,
        out_channels,
        base_channels,
        latent_dim,
        strides,
        dilations,
        residual_kernel_size,
    ):
        super().__init__()
        current_channels = base_channels * (2 ** len(strides))
        layers = [nn.Conv1d(latent_dim, current_channels, kernel_size=7, padding=3)]
        for stride in reversed(strides):
            layers.append(DecoderBlock(current_channels, stride, dilations, residual_kernel_size))
            current_channels //= 2
        layers.extend(
            [
                Snake1d(base_channels),
                nn.Conv1d(base_channels, out_channels, kernel_size=7, padding=3),
                nn.Tanh(),
            ]
        )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class QuantizerOutput:
    def __init__(self, quantized, commitment_loss, codebook_loss, perplexity, codes):
        self.quantized = quantized
        self.commitment_loss = commitment_loss
        self.codebook_loss = codebook_loss
        self.perplexity = perplexity
        self.codes = codes
