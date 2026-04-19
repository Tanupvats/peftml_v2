"""PACT — Parameterized Clipping Activation (Choi et al., 2018).

Replaces ReLU / ReLU6 activations with a learnable clipping bound ``α``
so that the activation range is co-optimised with the rest of the network
during quantization-aware training.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..core.utils import get_logger, set_attr_by_name
from .ste import round_ste

logger = get_logger(__name__)

__all__ = ["PACTReLU", "replace_with_pact"]


class PACTReLU(nn.Module):
    """Learnable-clipping activation for unsigned activation quantization.

    During QAT the activation is clipped to ``[0, α]`` and uniformly
    quantized to *bits* unsigned levels.  ``α`` is a trainable parameter.
    """

    def __init__(self, bits: int = 8, alpha_init: float = 6.0) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.bits = bits
        self.qmax = (2**bits) - 1  # unsigned

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Differentiable clamp to [0, alpha]
        x_clipped = 0.5 * (x.abs() - (x - self.alpha).abs() + self.alpha)

        scale = torch.clamp(self.alpha / self.qmax, min=1e-8)
        return round_ste(x_clipped / scale) * scale

    def extra_repr(self) -> str:
        return f"bits={self.bits}, alpha_init={self.alpha.item():.3f}"


def replace_with_pact(
    model: nn.Module,
    bits: int = 8,
    alpha_init: float = 6.0,
    ignore_names: Optional[list[str]] = None,
) -> nn.Module:
    """Replace ``nn.ReLU`` / ``nn.ReLU6`` with :class:`PACTReLU` in-place."""
    ignore_names = ignore_names or []
    replacements: list[tuple[str, nn.Module]] = []

    for name, module in model.named_modules():
        if any(ign in name for ign in ignore_names):
            continue
        if isinstance(module, (nn.ReLU, nn.ReLU6)):
            replacements.append((name, PACTReLU(bits=bits, alpha_init=alpha_init)))

    for name, new_module in replacements:
        set_attr_by_name(model, name, new_module)

    logger.info("Replaced %d activations with PACT-%dbit.", len(replacements), bits)
    return model
