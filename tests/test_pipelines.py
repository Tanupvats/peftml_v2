"""Integration tests for peftml pipelines."""

import copy

import torch
import torch.nn as nn
import pytest

from peftml import ModelCompressor, SparseQATPipeline
from peftml.core.config import SparseQATConfig, TaskType
from peftml.core.utils import count_parameters


class TestModelCompressor:
    def test_prune_global(self, tiny_cnn, cnn_batch):
        comp = ModelCompressor(tiny_cnn)
        pruner = comp.prune(method="global", amount=0.3)
        assert pruner.is_pruned()

        images, labels = cnn_batch
        out = tiny_cnn(images)
        assert out.shape == (4, 10)

        pruner.commit()
        assert not pruner.is_pruned()

    def test_prune_unstructured(self, tiny_cnn):
        comp = ModelCompressor(tiny_cnn)
        pruner = comp.prune(method="unstructured", amount=0.2)
        assert pruner.is_pruned()

    def test_prune_structured(self, tiny_cnn):
        comp = ModelCompressor(tiny_cnn)
        pruner = comp.prune(method="structured", amount=0.1)
        assert pruner.is_pruned()

    def test_prune_invalid_method(self, tiny_cnn):
        comp = ModelCompressor(tiny_cnn)
        with pytest.raises(ValueError):
            comp.prune(method="magic", amount=0.3)

    def test_quantize_for_qat(self, tiny_cnn, cnn_batch):
        comp = ModelCompressor(tiny_cnn)
        model = comp.quantize_for_qat(bits=8)
        images, _ = cnn_batch
        out = model(images)
        assert out.shape == (4, 10)

    def test_distill(self, tiny_cnn, cnn_batch):
        teacher = copy.deepcopy(tiny_cnn)
        comp = ModelCompressor(tiny_cnn)
        trainer = comp.distill(teacher, task_type="classification")

        images, labels = cnn_batch
        criterion = nn.CrossEntropyLoss()
        loss, out = trainer(images, labels, criterion)
        assert loss.item() > 0
        assert out.shape == (4, 10)

    def test_apply_lora(self, tiny_llm, llm_batch):
        comp = ModelCompressor(tiny_llm)
        model = comp.apply_lora(target_modules=["q_proj", "v_proj"], r=4)
        out = model(llm_batch)
        assert out.shape == (2, 16, 256)

        stats = count_parameters(model)
        assert stats["trainable_pct"] < 20  # most params should be frozen

    def test_apply_qlora(self, tiny_llm, llm_batch):
        comp = ModelCompressor(tiny_llm)
        model = comp.apply_qlora(target_modules=["q_proj", "v_proj"], r=4)
        out = model(llm_batch)
        assert out.shape == (2, 16, 256)

    def test_merge_lora(self, tiny_llm, llm_batch):
        comp = ModelCompressor(tiny_llm)
        comp.apply_lora(target_modules=["q_proj", "v_proj"], r=4)

        tiny_llm.eval()
        with torch.no_grad():
            out_before = tiny_llm(llm_batch)

        comp.merge_lora()
        with torch.no_grad():
            out_after = tiny_llm(llm_batch)

        assert torch.allclose(out_before, out_after, atol=1e-5)

    def test_merge_lora_without_injection_raises(self, tiny_llm):
        comp = ModelCompressor(tiny_llm)
        with pytest.raises(RuntimeError, match="No LoRA"):
            comp.merge_lora()

    def test_get_lora_state_dict(self, tiny_llm):
        comp = ModelCompressor(tiny_llm)
        comp.apply_lora(target_modules=["q_proj", "v_proj"], r=4)
        sd = comp.get_lora_state_dict()
        assert len(sd) > 0


class TestSparseQATPipeline:
    def test_train_step(self, tiny_cnn, cnn_batch):
        config = SparseQATConfig(
            task_type=TaskType.CLASSIFICATION,
            bits=8,
            target_sparsity=0.3,
            pruning_steps=3,
        )
        pipe = SparseQATPipeline(tiny_cnn, config=config)
        optimizer = torch.optim.SGD(tiny_cnn.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        loss = pipe.train_step(cnn_batch, optimizer, criterion, device="cpu")
        assert loss > 0

    def test_epoch_stepping(self, tiny_cnn, cnn_batch):
        config = SparseQATConfig(
            task_type=TaskType.CLASSIFICATION,
            bits=8,
            target_sparsity=0.4,
            pruning_steps=3,
        )
        pipe = SparseQATPipeline(tiny_cnn, config=config)
        optimizer = torch.optim.SGD(tiny_cnn.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        for _ in range(3):
            pipe.train_step(cnn_batch, optimizer, criterion, device="cpu")
            stats = pipe.step_epoch()
            assert "global_sparsity_pct" in stats

    def test_export(self, tiny_cnn, cnn_batch):
        config = SparseQATConfig(bits=8, target_sparsity=0.3, pruning_steps=2)
        pipe = SparseQATPipeline(tiny_cnn, config=config)
        optimizer = torch.optim.SGD(tiny_cnn.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        for _ in range(2):
            pipe.train_step(cnn_batch, optimizer, criterion, device="cpu")
            pipe.step_epoch()

        model = pipe.export()
        model.eval()
        images, _ = cnn_batch
        with torch.no_grad():
            out = model(images)
        assert out.shape == (4, 10)


class TestCoreUtilities:
    def test_count_parameters(self, tiny_cnn):
        stats = count_parameters(tiny_cnn)
        assert stats["total"] > 0
        assert stats["trainable"] == stats["total"]
        assert stats["frozen"] == 0

    def test_count_after_freeze(self, tiny_cnn):
        from peftml import freeze

        freeze(tiny_cnn)
        stats = count_parameters(tiny_cnn)
        assert stats["trainable"] == 0
        assert stats["frozen"] == stats["total"]
