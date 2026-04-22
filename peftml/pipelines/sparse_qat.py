

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ..core.config import SparseQATConfig, TaskType
from ..core.utils import get_logger
from ..pruning.pruner import DynamicPruner
from ..pruning.schedulers import IterativePruningScheduler
from ..quantization.lsq import replace_with_lsq
from ..quantization.pact import replace_with_pact

logger = get_logger(__name__)

__all__ = ["SparseQATPipeline"]


class SparseQATPipeline:
    """End-to-end pruning + QAT pipeline.

    Usage::

        pipe = SparseQATPipeline(model, config=SparseQATConfig(bits=8, target_sparsity=0.5))
        for epoch in range(epochs):
            for batch in loader:
                loss = pipe.train_step(batch, optimizer, criterion)
            pipe.step_epoch()
        model = pipe.export()

    Parameters
    ----------
    model : nn.Module
        The model to compress.
    config : SparseQATConfig
        Pipeline configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[SparseQATConfig] = None,
    ) -> None:
        self.config = config or SparseQATConfig()

        # Step 1: Inject quantisation nodes FIRST
        logger.info("Injecting LSQ weight quantisation (INT%d)...", self.config.bits)
        self.model = replace_with_lsq(model, bits=self.config.bits, ignore_names=self.config.ignore_layers)

        logger.info("Injecting PACT activation quantisation (INT%d)...", self.config.bits)
        self.model = replace_with_pact(self.model, bits=self.config.bits, ignore_names=self.config.ignore_layers)

        # Step 2: Attach pruning masks on top of the QAT graph
        logger.info("Initialising pruner on QAT graph...")
        self.pruner = DynamicPruner(self.model, ignore_layers=self.config.ignore_layers)

        # Step 3: Iterative pruning scheduler
        self.scheduler = IterativePruningScheduler(
            pruner=self.pruner,
            target_sparsity=self.config.target_sparsity,
            total_steps=self.config.pruning_steps,
        )

        self.model.train()
        self._epoch = 0

    # -----------------------------------------------------------------
    # Training interface
    # -----------------------------------------------------------------
    def train_step(
        self,
        batch: Any,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = "cuda",
    ) -> float:
        """Execute one forward/backward step.  Returns the scalar loss."""
        self.model.to(device)
        self.model.train()
        optimizer.zero_grad()

        task = self.config.task_type

        if task in (TaskType.CLASSIFICATION, TaskType.SEGMENTATION):
            inputs, targets = batch[0].to(device), batch[1].to(device)
            outputs = self.model(inputs)
            if isinstance(outputs, dict) and "out" in outputs:
                outputs = outputs["out"]
            loss = criterion(outputs, targets)

        elif task == TaskType.DETECTION:
            images = [img.to(device) for img in batch[0]]
            targets = [{k: v.to(device) for k, v in t.items()} for t in batch[1]]
            loss_dict = self.model(images, targets)
            loss = sum(loss_dict.values())

        elif task == TaskType.LANGUAGE_MODELING:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = self.model(input_ids)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))

        else:
            raise ValueError(f"Unsupported task_type: {task}")

        loss.backward()
        optimizer.step()
        return loss.item()

    def step_epoch(self) -> Dict[str, float]:
        """Call at the end of each epoch to update the pruning schedule.

        Returns current sparsity statistics.
        """
        self._epoch += 1
        self.scheduler.step()
        stats = self.pruner.compute_sparsity()
        logger.info(
            "Epoch %d — global sparsity: %.2f%%",
            self._epoch,
            stats["global_sparsity_pct"],
        )
        return stats

    # -----------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------
    def export(self) -> nn.Module:
        """Finalise: bake masks and switch to eval mode."""
        self.model.eval()
        self.pruner.commit()
        stats = self.pruner.compute_sparsity()
        logger.info(
            "Export complete — INT%d, %.1f%% sparse.",
            self.config.bits,
            stats["global_sparsity_pct"],
        )
        return self.model
