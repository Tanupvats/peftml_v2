"""Shared pytest fixtures for peftml tests."""

import pytest
import torch
import torch.nn as nn


class TinyCNN(nn.Module):
    """Minimal CNN for fast unit tests."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


class TinyTransformerBlock(nn.Module):
    """Minimal transformer-like block with q_proj / v_proj naming."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        h = self.norm(x)
        q, k, v = self.q_proj(h), self.k_proj(h), self.v_proj(h)
        # Simplified attention (no masking / multi-head for speed)
        attn = torch.softmax(q @ k.transpose(-1, -2) / (q.size(-1) ** 0.5), dim=-1)
        h = self.out_proj(attn @ v)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class TinyLLM(nn.Module):
    """Minimal LLM-like model for LoRA tests."""

    def __init__(self, dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(256, dim)
        self.layers = nn.ModuleList([TinyTransformerBlock(dim) for _ in range(num_layers)])
        self.head = nn.Linear(dim, 256)

    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(h)


@pytest.fixture
def tiny_cnn():
    return TinyCNN(num_classes=10)


@pytest.fixture
def tiny_llm():
    return TinyLLM(dim=64, num_layers=2)


@pytest.fixture
def cnn_batch():
    """Random batch: (images, labels)."""
    return torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,))


@pytest.fixture
def llm_batch():
    """Random token batch."""
    return torch.randint(0, 256, (2, 16))
