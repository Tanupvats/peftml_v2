"""Tests for peftml.distillation."""

import torch
import torch.nn as nn
import pytest

from peftml.distillation.losses import (
    hinton_kd_loss,
    spatial_feature_loss,
    cosine_feature_loss,
    attention_transfer_loss,
)
from peftml.distillation.adapters import ChannelAdapter, LinearAdapter
from peftml.distillation.trainer import DynamicKDTrainer
from peftml.core.config import KDConfig, FeatureMapping, TaskType


class TestLosses:
    def test_hinton_kd_loss(self):
        s = torch.randn(4, 10)
        t = torch.randn(4, 10)
        loss = hinton_kd_loss(s, t, temperature=4.0)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_spatial_feature_loss(self):
        s = torch.randn(2, 16, 8, 8)
        t = torch.randn(2, 16, 8, 8)
        loss = spatial_feature_loss(s, t)
        assert loss.ndim == 0

    def test_cosine_feature_loss(self):
        s = torch.randn(2, 16, 8, 8)
        t = torch.randn(2, 16, 8, 8)
        loss = cosine_feature_loss(s, t)
        assert loss.ndim == 0

    def test_attention_transfer_loss(self):
        s = torch.randn(2, 16, 8, 8)
        t = torch.randn(2, 16, 8, 8)
        loss = attention_transfer_loss(s, t)
        assert loss.ndim == 0


class TestAdapters:
    def test_channel_adapter(self):
        adapter = ChannelAdapter(16, 64)
        x = torch.randn(2, 16, 8, 8)
        y = adapter(x)
        assert y.shape == (2, 64, 8, 8)

    def test_linear_adapter(self):
        adapter = LinearAdapter(32, 128)
        x = torch.randn(2, 10, 32)
        y = adapter(x)
        assert y.shape == (2, 10, 128)


class TestDynamicKDTrainer:
    def test_classification(self, tiny_cnn):
        """Student and teacher are the same arch — simplest test."""
        import copy

        teacher = copy.deepcopy(tiny_cnn)
        config = KDConfig(task_type=TaskType.CLASSIFICATION, temperature=4.0, alpha=0.5)
        trainer = DynamicKDTrainer(student=tiny_cnn, teacher=teacher, config=config)

        images = torch.randn(4, 3, 32, 32)
        labels = torch.randint(0, 10, (4,))
        criterion = nn.CrossEntropyLoss()

        loss, out = trainer(images, labels, criterion)
        assert loss.ndim == 0
        assert out.shape == (4, 10)

        # Verify teacher is frozen
        for p in teacher.parameters():
            assert not p.requires_grad

    def test_with_feature_mapping(self, tiny_cnn):
        import copy

        teacher = copy.deepcopy(tiny_cnn)
        fm = FeatureMapping(
            student_layer="features.0",  # first conv
            teacher_layer="features.0",
            student_channels=16,
            teacher_channels=16,
        )
        config = KDConfig(
            task_type=TaskType.CLASSIFICATION,
            feature_mappings=[fm],
            beta=10.0,
        )
        trainer = DynamicKDTrainer(student=tiny_cnn, teacher=teacher, config=config)

        images = torch.randn(4, 3, 32, 32)
        labels = torch.randint(0, 10, (4,))
        criterion = nn.CrossEntropyLoss()

        loss, out = trainer(images, labels, criterion)
        assert loss.item() > 0

    def test_teardown_removes_hooks(self, tiny_cnn):
        import copy

        teacher = copy.deepcopy(tiny_cnn)
        fm = FeatureMapping("features.0", "features.0", 16, 16)
        config = KDConfig(feature_mappings=[fm])
        trainer = DynamicKDTrainer(student=tiny_cnn, teacher=teacher, config=config)
        assert len(trainer._hooks) == 2
        trainer.teardown()
        assert len(trainer._hooks) == 0

    def test_invalid_layer_raises(self, tiny_cnn):
        import copy

        teacher = copy.deepcopy(tiny_cnn)
        fm = FeatureMapping("nonexistent_layer", "features.0", 16, 16)
        config = KDConfig(feature_mappings=[fm])
        with pytest.raises(ValueError, match="not found"):
            DynamicKDTrainer(student=tiny_cnn, teacher=teacher, config=config)
