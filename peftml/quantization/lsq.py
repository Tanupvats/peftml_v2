"""Learned Step-size Quantization (LSQ) for weights.

Implements the LSQ method from Esser et al., 2020.  Provides drop-in
replacements for ``nn.Conv2d`` and ``nn.Linear`` that learn a per-tensor
(or per-channel) step size during QAT.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.config import LSQConfig
from ..core.utils import get_logger, get_modules_by_type, set_attr_by_name
from .ste import grad_scale, round_ste

logger = get_logger(__name__)

__all__ = ["QConv2d", "QLinear", "replace_with_lsq"]


def _init_scale(weight: torch.Tensor, qmax: int) -> float:
    """LSQ heuristic: ``2 * E[|W|] / sqrt(Q_max)``."""
    return float(2.0 * weight.detach().abs().mean() / math.sqrt(qmax))


# ---------------------------------------------------------------------------
# Quantised Conv2d
# ---------------------------------------------------------------------------
class QConv2d(nn.Conv2d):
    """Drop-in ``nn.Conv2d`` with LSQ weight quantization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        bits: int = 8,
        **kwargs,
    ):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.bits = bits
        self.qmax = (2 ** (bits - 1)) - 1
        self.qmin = -(2 ** (bits - 1))
        self.step_size = nn.Parameter(torch.tensor(1.0))
        self._initialized = False

    def _lazy_init(self) -> None:
        self.step_size.data.fill_(_init_scale(self.weight, self.qmax))
        self._initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._initialized:
            self._lazy_init()

        g = 1.0 / math.sqrt(self.weight.numel() * self.qmax)
        s = F.softplus(grad_scale(self.step_size, g)) + 1e-8

        w_q = round_ste(self.weight / s).clamp(self.qmin, self.qmax) * s

        return F.conv2d(
            x, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


# ---------------------------------------------------------------------------
# Quantised Linear
# ---------------------------------------------------------------------------
class QLinear(nn.Linear):
    """Drop-in ``nn.Linear`` with LSQ weight quantization."""

    def __init__(self, in_features: int, out_features: int, bits: int = 8, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias)
        self.bits = bits
        self.qmax = (2 ** (bits - 1)) - 1
        self.qmin = -(2 ** (bits - 1))
        self.step_size = nn.Parameter(torch.tensor(1.0))
        self._initialized = False

    def _lazy_init(self) -> None:
        self.step_size.data.fill_(_init_scale(self.weight, self.qmax))
        self._initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._initialized:
            self._lazy_init()

        g = 1.0 / math.sqrt(self.weight.numel() * self.qmax)
        s = F.softplus(grad_scale(self.step_size, g)) + 1e-8

        w_q = round_ste(self.weight / s).clamp(self.qmin, self.qmax) * s

        return F.linear(x, w_q, self.bias)


# ---------------------------------------------------------------------------
# Recursive replacement helper
# ---------------------------------------------------------------------------
def replace_with_lsq(
    model: nn.Module,
    bits: int = 8,
    ignore_names: Optional[list[str]] = None,
) -> nn.Module:
    """Recursively replace ``nn.Conv2d`` / ``nn.Linear`` with their LSQ counterparts.

    Parameters
    ----------
    model:
        The model to modify **in-place**.
    bits:
        Bit-width for weight quantization.
    ignore_names:
        Substrings — any module whose fully-qualified name contains one of
        these strings will be skipped (e.g. ``["classifier", "head"]``).

    Returns
    -------
    nn.Module
        The same model reference, modified in-place.
    """
    ignore_names = ignore_names or []
    replacements: list[tuple[str, nn.Module]] = []

    for name, module in model.named_modules():
        if any(ign in name for ign in ignore_names):
            continue

        if isinstance(module, nn.Conv2d) and not isinstance(module, QConv2d):
            q = QConv2d(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                bits=bits,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None,
            )
            q.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                q.bias.data.copy_(module.bias.data)
            replacements.append((name, q))

        elif isinstance(module, nn.Linear) and not isinstance(module, QLinear):
            q = QLinear(
                module.in_features,
                module.out_features,
                bits=bits,
                bias=module.bias is not None,
            )
            q.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                q.bias.data.copy_(module.bias.data)
            replacements.append((name, q))

    for name, new_module in replacements:
        set_attr_by_name(model, name, new_module)

    logger.info("Replaced %d layers with LSQ-%dbit variants.", len(replacements), bits)
    return model
