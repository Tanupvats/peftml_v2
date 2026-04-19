"""Tests for peftml.lora."""

import torch
import torch.nn as nn
import pytest

from peftml.lora.layers import LoRALinear
from peftml.lora.orchestrator import QLoRAOrchestrator
from peftml.core.config import LoRAConfig, QLoRAConfig


class TestLoRALinear:
    def test_output_matches_base_at_init(self):
        """B=0 at init, so LoRA output should equal the base output."""
        base = nn.Linear(32, 16)
        lora = LoRALinear(base, r=4, lora_alpha=8)
        x = torch.randn(2, 32)

        with torch.no_grad():
            base_out = base(x)
            lora_out = lora(x)
        assert torch.allclose(base_out, lora_out, atol=1e-6)

    def test_only_adapter_requires_grad(self):
        base = nn.Linear(32, 16)
        lora = LoRALinear(base, r=4)
        trainable = [n for n, p in lora.named_parameters() if p.requires_grad]
        assert "lora_A" in trainable
        assert "lora_B" in trainable
        assert "base_layer.weight" not in trainable

    def test_merge_changes_base_weight(self):
        base = nn.Linear(16, 8)
        lora = LoRALinear(base, r=2, lora_alpha=4)
        # Manually set A and B to non-zero
        lora.lora_A.data.fill_(0.1)
        lora.lora_B.data.fill_(0.1)
        weight_before = base.weight.data.clone()
        lora.merge()
        assert not torch.allclose(base.weight.data, weight_before)
        assert lora.merged

    def test_unmerge_restores_weight(self):
        base = nn.Linear(16, 8)
        lora = LoRALinear(base, r=2, lora_alpha=4)
        lora.lora_A.data.fill_(0.1)
        lora.lora_B.data.fill_(0.1)
        weight_before = base.weight.data.clone()
        lora.merge()
        lora.unmerge()
        assert torch.allclose(base.weight.data, weight_before, atol=1e-6)

    def test_forward_shape(self):
        lora = LoRALinear(nn.Linear(64, 32), r=8)
        x = torch.randn(4, 64)
        assert lora(x).shape == (4, 32)

    def test_gradient_flows_through_adapter(self):
        lora = LoRALinear(nn.Linear(16, 8), r=4)
        x = torch.randn(2, 16)
        y = lora(x).sum()
        y.backward()
        assert lora.lora_A.grad is not None
        assert lora.lora_B.grad is not None


class TestQLoRAOrchestrator:
    def test_injects_adapters(self, tiny_llm):
        config = LoRAConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"])
        orch = QLoRAOrchestrator(tiny_llm, config=config)
        orch.prepare()

        lora_count = sum(1 for m in tiny_llm.modules() if isinstance(m, LoRALinear))
        # 2 layers * 2 targets = 4
        assert lora_count == 4

    def test_base_frozen(self, tiny_llm):
        config = LoRAConfig(target_modules=["q_proj", "v_proj"])
        orch = QLoRAOrchestrator(tiny_llm, config=config)
        orch.prepare()

        for name, p in tiny_llm.named_parameters():
            if "lora_" not in name and "norm" not in name:
                assert not p.requires_grad, f"{name} should be frozen"

    def test_forward_works(self, tiny_llm, llm_batch):
        config = LoRAConfig(target_modules=["q_proj", "v_proj"])
        orch = QLoRAOrchestrator(tiny_llm, config=config)
        orch.prepare()
        out = tiny_llm(llm_batch)
        assert out.shape == (2, 16, 256)

    def test_merge_and_unmerge(self, tiny_llm, llm_batch):
        config = LoRAConfig(target_modules=["q_proj", "v_proj"])
        orch = QLoRAOrchestrator(tiny_llm, config=config)
        orch.prepare()

        # Get reference output
        tiny_llm.eval()
        with torch.no_grad():
            out_before = tiny_llm(llm_batch)

        orch.merge_adapters()
        with torch.no_grad():
            out_merged = tiny_llm(llm_batch)

        # Merged output should match (within numerical tolerance)
        assert torch.allclose(out_before, out_merged, atol=1e-5)

    def test_adapter_state_dict(self, tiny_llm):
        config = LoRAConfig(target_modules=["q_proj", "v_proj"])
        orch = QLoRAOrchestrator(tiny_llm, config=config)
        orch.prepare()

        sd = orch.get_adapter_state_dict()
        assert len(sd) > 0
        assert all("lora_" in k for k in sd)

    def test_qlora_stabilises_norms(self, tiny_llm):
        config = QLoRAConfig(target_modules=["q_proj", "v_proj"])
        orch = QLoRAOrchestrator(tiny_llm, config=config)
        orch.prepare()

        for m in tiny_llm.modules():
            if isinstance(m, nn.LayerNorm):
                assert m.weight.dtype == torch.float32
