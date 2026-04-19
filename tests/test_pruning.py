"""Tests for peftml.pruning."""

import torch
import pytest

from peftml.pruning import DynamicPruner, IterativePruningScheduler


class TestDynamicPruner:
    def test_global_creates_masks(self, tiny_cnn):
        pruner = DynamicPruner(tiny_cnn)
        pruner.apply_global(amount=0.3)
        assert pruner.is_pruned()

    def test_unstructured(self, tiny_cnn):
        pruner = DynamicPruner(tiny_cnn)
        pruner.apply_unstructured(amount=0.2)
        stats = pruner.compute_sparsity()
        assert stats["global_sparsity_pct"] > 0

    def test_structured(self, tiny_cnn):
        pruner = DynamicPruner(tiny_cnn)
        pruner.apply_structured(amount=0.1)
        assert pruner.is_pruned()

    def test_ignore_layers(self, tiny_cnn):
        pruner = DynamicPruner(tiny_cnn, ignore_layers=["classifier"])
        pruner.apply_global(amount=0.5)
        # classifier weight should be untouched
        assert not hasattr(tiny_cnn.classifier, "weight_mask")

    def test_commit_removes_masks(self, tiny_cnn):
        pruner = DynamicPruner(tiny_cnn)
        pruner.apply_global(amount=0.3)
        assert pruner.is_pruned()
        pruner.commit()
        assert not pruner.is_pruned()

    def test_sparsity_within_tolerance(self, tiny_cnn):
        pruner = DynamicPruner(tiny_cnn)
        pruner.apply_global(amount=0.4)
        stats = pruner.compute_sparsity()
        # Global pruning should be close to the target
        assert 35 < stats["global_sparsity_pct"] < 45

    def test_forward_after_pruning(self, tiny_cnn, cnn_batch):
        pruner = DynamicPruner(tiny_cnn)
        pruner.apply_global(amount=0.3)
        images, _ = cnn_batch
        out = tiny_cnn(images)
        assert out.shape == (4, 10)


class TestIterativePruningScheduler:
    def test_monotonic_sparsity(self, tiny_cnn):
        pruner = DynamicPruner(tiny_cnn)
        sched = IterativePruningScheduler(pruner, target_sparsity=0.5, total_steps=5)

        targets = []
        for _ in range(5):
            t = sched.step()
            targets.append(t)

        # Targets should be monotonically non-decreasing
        for i in range(1, len(targets)):
            assert targets[i] >= targets[i - 1]

    def test_final_sparsity_near_target(self, tiny_cnn):
        pruner = DynamicPruner(tiny_cnn)
        target = 0.5
        sched = IterativePruningScheduler(pruner, target_sparsity=target, total_steps=10)

        for _ in range(10):
            sched.step()

        stats = pruner.compute_sparsity()
        assert stats["global_sparsity_pct"] > 40  # should be close to 50%

    def test_finished_flag(self, tiny_cnn):
        pruner = DynamicPruner(tiny_cnn)
        sched = IterativePruningScheduler(pruner, target_sparsity=0.3, total_steps=3)
        assert not sched.finished
        for _ in range(3):
            sched.step()
        assert sched.finished

    def test_validation_errors(self, tiny_cnn):
        pruner = DynamicPruner(tiny_cnn)
        with pytest.raises(ValueError):
            IterativePruningScheduler(pruner, target_sparsity=1.5, total_steps=5)
        with pytest.raises(ValueError):
            IterativePruningScheduler(pruner, target_sparsity=0.5, total_steps=0)
