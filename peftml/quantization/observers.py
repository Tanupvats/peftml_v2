

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..core.utils import get_logger

logger = get_logger(__name__)

__all__ = ["ActivationObserver"]


class ActivationObserver:
    """Hooks into a model to record per-channel activation maxima.

    The observer correctly handles both 4-D (NCHW — convolutions) and
    2-D (BF — linear layers) activations, which the original implementation
    failed to do.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self.activation_max: Dict[str, torch.Tensor] = {}
        self.activation_mean: Dict[str, torch.Tensor] = {}
        self._call_count: Dict[str, int] = {}

    # -----------------------------------------------------------------
    # Internal hook
    # -----------------------------------------------------------------
    def _make_hook(self, name: str):
        def hook(_module: nn.Module, _input, output):
            if not isinstance(output, torch.Tensor):
                return  # skip dict / tuple outputs (detection heads, etc.)

            t = output.detach()
            ndim = t.ndim

            # Reduce over all dims except channels.
            # Conv2d → (N, C, H, W) → reduce (0, 2, 3)
            # Linear → (*, F)        → reduce all but last
            if ndim == 4:
                reduce_dims = (0, 2, 3)
            elif ndim == 3:
                reduce_dims = (0, 1)  # (B, T, F) — e.g. transformer
            elif ndim == 2:
                reduce_dims = (0,)    # (B, F)
            else:
                return  # unusual shape — skip

            batch_max = t.abs().amax(dim=reduce_dims)
            batch_mean = t.abs().mean(dim=reduce_dims)

            if name not in self.activation_max:
                self.activation_max[name] = batch_max
                self.activation_mean[name] = batch_mean
                self._call_count[name] = 1
            else:
                self.activation_max[name] = torch.max(self.activation_max[name], batch_max)
                # Running mean
                n = self._call_count[name]
                self.activation_mean[name] = (self.activation_mean[name] * n + batch_mean) / (n + 1)
                self._call_count[name] = n + 1

        return hook

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def attach(self) -> "ActivationObserver":
        """Register hooks on all Conv2d / Linear layers."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self._hooks.append(module.register_forward_hook(self._make_hook(name)))
        logger.info("Attached activation observers to %d layers.", len(self._hooks))
        return self

    def remove(self) -> None:
        """Remove all hooks (call after calibration)."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        logger.info("Removed activation observers.")

    @torch.no_grad()
    def calibrate(
        self,
        dataloader,
        device: str = "cuda",
        max_batches: Optional[int] = None,
    ) -> None:
        """Run calibration by forwarding *max_batches* through the model.

        Handles dataloaders that yield ``(input, target)`` tuples or bare
        tensors.
        """
        self.model.to(device)
        self.model.eval()
        self.attach()

        for idx, batch in enumerate(dataloader):
            if max_batches is not None and idx >= max_batches:
                break
            inputs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            self.model(inputs)

        self.remove()
        logger.info("Calibration complete (%d batches).", idx + 1 if 'idx' in dir() else 0)
