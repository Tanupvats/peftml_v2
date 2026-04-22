

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from ..core.config import PruningConfig, PruningMethod
from ..core.utils import get_logger, get_modules_by_type

logger = get_logger(__name__)

__all__ = ["DynamicPruner"]


class DynamicPruner:
    """Unified pruning manager.

    Collects ``nn.Conv2d`` and ``nn.Linear`` layers (skipping BatchNorm and
    user-specified layers) and applies one of three pruning strategies.
    """

    def __init__(
        self,
        model: nn.Module,
        ignore_layers: Optional[List[str]] = None,
    ) -> None:
        self.model = model
        self.ignore_layers = ignore_layers or []

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------
    def _prunable_modules(self) -> List[Tuple[str, nn.Module]]:
        return get_modules_by_type(
            self.model, (nn.Conv2d, nn.Linear), ignore_names=self.ignore_layers
        )

    # -----------------------------------------------------------------
    # Pruning strategies
    # -----------------------------------------------------------------
    def apply_unstructured(self, amount: float = 0.2) -> None:
        """Per-layer L1-magnitude unstructured pruning."""
        modules = self._prunable_modules()
        for name, module in modules:
            prune.l1_unstructured(module, name="weight", amount=amount)
        logger.info("Unstructured L1 pruning (%.0f%% per layer) on %d layers.", amount * 100, len(modules))

    def apply_structured(self, amount: float = 0.1, n: int = 2, dim: int = 0) -> None:
        """Per-layer Ln-structured pruning along *dim* (default: output filters)."""
        modules = self._prunable_modules()
        for name, module in modules:
            prune.ln_structured(module, name="weight", amount=amount, n=n, dim=dim)
        logger.info(
            "Structured L%d pruning (%.0f%%, dim=%d) on %d layers.", n, amount * 100, dim, len(modules)
        )

    def apply_global(self, amount: float = 0.2) -> None:
        """Global L1-unstructured pruning across the entire network."""
        params = [(m, "weight") for _, m in self._prunable_modules()]
        if not params:
            logger.warning("No prunable modules found.")
            return
        prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=amount)
        logger.info("Global unstructured pruning (%.0f%% total).", amount * 100)

    def apply(self, config: PruningConfig) -> None:
        """Apply pruning from a :class:`PruningConfig`."""
        self.ignore_layers = config.ignore_layers or self.ignore_layers
        dispatch = {
            PruningMethod.GLOBAL: self.apply_global,
            PruningMethod.UNSTRUCTURED: self.apply_unstructured,
            PruningMethod.STRUCTURED: self.apply_structured,
        }
        dispatch[config.method](amount=config.amount)

    # -----------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------
    def compute_sparsity(self) -> Dict[str, object]:
        """Return global and per-layer sparsity percentages."""
        total_zeros = 0
        total_elems = 0
        layer_sparsity: Dict[str, float] = {}

        for name, module in self._prunable_modules():
            w = module.weight
            zeros = float((w == 0).sum())
            elems = float(w.nelement())
            total_zeros += zeros
            total_elems += elems
            if elems > 0:
                layer_sparsity[name] = zeros / elems * 100

        global_pct = (total_zeros / total_elems * 100) if total_elems > 0 else 0.0
        return {"global_sparsity_pct": global_pct, "layer_sparsity_pct": layer_sparsity}

    # -----------------------------------------------------------------
    # Finalization
    # -----------------------------------------------------------------
    def commit(self) -> None:
        """Make pruning permanent — bake masks into weights.

        Must be called before saving the model for deployment.
        """
        committed = 0
        for _name, module in self._prunable_modules():
            try:
                prune.remove(module, "weight")
                committed += 1
            except ValueError:
                pass  # not pruned
        logger.info("Committed pruning masks on %d layers.", committed)

    def is_pruned(self) -> bool:
        """Return ``True`` if any prunable layer still has a reparameterization mask."""
        return any(hasattr(m, "weight_mask") for _, m in self._prunable_modules())
