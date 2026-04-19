"""Knowledge distillation: losses, feature adapters, and the unified trainer."""

from .adapters import ChannelAdapter, LinearAdapter
from .losses import (
    attention_transfer_loss,
    cosine_feature_loss,
    hinton_kd_loss,
    spatial_feature_loss,
)
from .trainer import DynamicKDTrainer

__all__ = [
    "ChannelAdapter",
    "DynamicKDTrainer",
    "LinearAdapter",
    "attention_transfer_loss",
    "cosine_feature_loss",
    "hinton_kd_loss",
    "spatial_feature_loss",
]
