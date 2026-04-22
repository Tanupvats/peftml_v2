

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["ChannelAdapter", "LinearAdapter"]


class ChannelAdapter(nn.Module):
    """1×1 convolution adapter for spatial feature maps (NCHW)."""

    def __init__(self, student_channels: int, teacher_channels: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(student_channels, teacher_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(teacher_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class LinearAdapter(nn.Module):
    """Linear projection adapter for 1-D feature vectors or transformer hidden states."""

    def __init__(self, student_dim: int, teacher_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(student_dim, teacher_dim, bias=False),
            nn.LayerNorm(teacher_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
