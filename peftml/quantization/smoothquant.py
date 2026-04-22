

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..core.utils import get_logger
from .observers import ActivationObserver

logger = get_logger(__name__)

__all__ = ["apply_smoothquant"]


def _find_preceding_layernorm(
    model: nn.Module,
    target_name: str,
) -> Optional[Tuple[str, nn.Module]]:
    """Heuristic: walk named_modules and return the LayerNorm/RMSNorm
    immediately preceding *target_name* in registration order.

    This is a best-effort helper — transformer architectures almost always
    have a norm right before each Linear projection.
    """
    prev: Optional[Tuple[str, nn.Module]] = None
    for name, module in model.named_modules():
        if name == target_name:
            return prev
        if isinstance(module, (nn.LayerNorm,)):
            prev = (name, module)
    return None


@torch.no_grad()
def apply_smoothquant(
    model: nn.Module,
    observer: ActivationObserver,
    alpha: float = 0.5,
) -> nn.Module:
    """Apply SmoothQuant channel equalization in-place.

    For each ``nn.Linear`` layer with recorded activation statistics the
    migration scale is::

        s_j = (max|X_j|^α) / (max|W_j|^(1-α))

    Weights are multiplied by ``s`` and — where possible — the preceding
    LayerNorm's weight/bias are divided by ``s`` so the mathematical output
    is unchanged.

    Parameters
    ----------
    model:
        The model to smooth (modified **in-place**).
    observer:
        A *calibrated* :class:`ActivationObserver`.
    alpha:
        Migration strength ∈ [0, 1].  Higher values push more difficulty
        onto the weights.

    Returns
    -------
    nn.Module
        The same model reference, smoothed in-place.
    """
    smoothed = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name not in observer.activation_max:
            continue

        x_max = observer.activation_max[name].clamp(min=1e-5)
        w_max = module.weight.abs().max(dim=0)[0].clamp(min=1e-5)

        # Per-channel migration scale
        scale = (x_max.pow(alpha)) / (w_max.pow(1.0 - alpha))

        # Scale weights:  W_new = W * diag(s)
        module.weight.data.mul_(scale.unsqueeze(0))

        # Attempt to absorb 1/s into the preceding LayerNorm
        prev = _find_preceding_layernorm(model, name)
        if prev is not None:
            ln_name, ln_module = prev
            if hasattr(ln_module, "weight") and ln_module.weight is not None:
                ln_module.weight.data.div_(scale)
            if hasattr(ln_module, "bias") and ln_module.bias is not None:
                ln_module.bias.data.div_(scale)

        smoothed += 1

    logger.info("SmoothQuant applied to %d Linear layers (alpha=%.2f).", smoothed, alpha)
    return model
