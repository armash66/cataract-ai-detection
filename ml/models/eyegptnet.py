"""Custom lightweight CNN for EyeGPT-AI (<5M params target)."""
from __future__ import annotations

import torch
from torch import nn


class DWSeparableBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class ResidualDSBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block1 = DWSeparableBlock(channels, channels, stride=1)
        self.block2 = DWSeparableBlock(channels, channels, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block2(self.block1(x))


class EyeGPTNet(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(DWSeparableBlock(32, 64, stride=2), ResidualDSBlock(64))
        self.stage2 = nn.Sequential(DWSeparableBlock(64, 128, stride=2), ResidualDSBlock(128))
        self.stage3 = nn.Sequential(DWSeparableBlock(128, 192, stride=2), ResidualDSBlock(192))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(192, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)
