"""Unified ``ModelCompressor`` — the top-level entry point.

Provides a single, fluent interface to every compression technique in
peftml.  Each method validates inputs, applies the transformation, and
returns either the modified model or a pipeline/trainer handle that the
caller drives through training.

Example::

    from peftml import ModelCompressor

    comp = ModelCompressor(my_resnet)

    # One-shot pruning
    pruner = comp.prune(method="global", amount=0.4)
    pruner.commit()

    # Or sparse QAT
    pipe = comp.sparse_qat(bits=8, target_sparsity=0.5)
    for epoch in range(30):
        for batch in loader:
            pipe.train_step(batch, optim, criterion)
        pipe.step_epoch()
    model = pipe.export()
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from ..core.config import (
    KDConfig,
    LoRAConfig,
    PruningConfig,
    PruningMethod,
    QLoRAConfig,
    SmoothQuantConfig,
    SparseQATConfig,
    TaskType,
)
from ..core.utils import get_logger
from ..distillation.trainer import DynamicKDTrainer
from ..lora.orchestrator import QLoRAOrchestrator
from ..pipelines.sparse_qat import SparseQATPipeline
from ..pruning.pruner import DynamicPruner
from ..quantization.lsq import replace_with_lsq
from ..quantization.observers import ActivationObserver
from ..quantization.pact import replace_with_pact
from ..quantization.smoothquant import apply_smoothquant

logger = get_logger(__name__)

__all__ = ["ModelCompressor"]


class ModelCompressor:
    """Unified facade for model compression.

    Parameters
    ----------
    model : nn.Module
        The model to compress.  Most methods modify it **in-place** and
        also return it (or a wrapper) for convenience.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    # ------------------------------------------------------------------
    # 1.  Pruning
    # ------------------------------------------------------------------
    def prune(
        self,
        method: str = "global",
        amount: float = 0.3,
        ignore_layers: Optional[List[str]] = None,
    ) -> DynamicPruner:
        """Apply one-shot pruning and return the :class:`DynamicPruner`.

        Call ``pruner.commit()`` before saving for deployment.
        """
        config = PruningConfig(
            method=PruningMethod(method),
            amount=amount,
            ignore_layers=ignore_layers or [],
        )
        pruner = DynamicPruner(self.model, ignore_layers=config.ignore_layers)
        pruner.apply(config)
        return pruner

    # ------------------------------------------------------------------
    # 2.  Quantization-Aware Training (LSQ + PACT)
    # ------------------------------------------------------------------
    def quantize_for_qat(
        self,
        bits: int = 8,
        ignore_layers: Optional[List[str]] = None,
    ) -> nn.Module:
        """Replace Conv2d/Linear with LSQ layers and ReLUs with PACT."""
        self.model = replace_with_lsq(self.model, bits=bits, ignore_names=ignore_layers)
        self.model = replace_with_pact(self.model, bits=bits, ignore_names=ignore_layers)
        return self.model

    # ------------------------------------------------------------------
    # 3.  Post-Training Quantization (SmoothQuant)
    # ------------------------------------------------------------------
    def apply_smoothquant(
        self,
        dataloader,
        alpha: float = 0.5,
        device: str = "cuda",
        calibration_batches: int = 8,
    ) -> nn.Module:
        """Calibrate and apply SmoothQuant in one call."""
        observer = ActivationObserver(self.model)
        observer.calibrate(dataloader, device=device, max_batches=calibration_batches)
        self.model = apply_smoothquant(self.model, observer, alpha=alpha)
        return self.model

    # ------------------------------------------------------------------
    # 4.  Sparse QAT (Pruning + Quantization)
    # ------------------------------------------------------------------
    def sparse_qat(
        self,
        task_type: str = "classification",
        bits: int = 8,
        target_sparsity: float = 0.5,
        pruning_steps: int = 30,
        ignore_layers: Optional[List[str]] = None,
    ) -> SparseQATPipeline:
        """Return a :class:`SparseQATPipeline` ready for training."""
        config = SparseQATConfig(
            task_type=TaskType(task_type),
            bits=bits,
            target_sparsity=target_sparsity,
            pruning_steps=pruning_steps,
            ignore_layers=ignore_layers or [],
        )
        return SparseQATPipeline(self.model, config=config)

    # ------------------------------------------------------------------
    # 5.  Knowledge Distillation
    # ------------------------------------------------------------------
    def distill(
        self,
        teacher: nn.Module,
        task_type: str = "classification",
        temperature: float = 4.0,
        alpha: float = 0.5,
        beta: float = 50.0,
        config: Optional[KDConfig] = None,
    ) -> DynamicKDTrainer:
        """Return a :class:`DynamicKDTrainer` wrapping student + teacher."""
        if config is None:
            config = KDConfig(
                task_type=TaskType(task_type),
                temperature=temperature,
                alpha=alpha,
                beta=beta,
            )
        return DynamicKDTrainer(student=self.model, teacher=teacher, config=config)

    # ------------------------------------------------------------------
    # 6.  LoRA / QLoRA
    # ------------------------------------------------------------------
    def apply_lora(
        self,
        target_modules: Optional[List[str]] = None,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ) -> nn.Module:
        """Inject standard LoRA adapters and freeze the base model."""
        config = LoRAConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules or ["q_proj", "v_proj"],
        )
        orch = QLoRAOrchestrator(self.model, config=config)
        self.model = orch.prepare()
        self._lora_orchestrator = orch
        return self.model

    def apply_qlora(
        self,
        target_modules: Optional[List[str]] = None,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ) -> nn.Module:
        """Inject QLoRA adapters with mixed-precision stability rules."""
        config = QLoRAConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules or ["q_proj", "v_proj"],
        )
        orch = QLoRAOrchestrator(self.model, config=config)
        self.model = orch.prepare()
        self._lora_orchestrator = orch
        return self.model

    def merge_lora(self) -> nn.Module:
        """Merge LoRA adapters into base weights for deployment."""
        if not hasattr(self, "_lora_orchestrator"):
            raise RuntimeError("No LoRA adapters have been injected yet.")
        return self._lora_orchestrator.merge_adapters()

    def get_lora_state_dict(self) -> dict:
        """Extract adapter-only state dict for lightweight saving."""
        if not hasattr(self, "_lora_orchestrator"):
            raise RuntimeError("No LoRA adapters have been injected yet.")
        return self._lora_orchestrator.get_adapter_state_dict()
