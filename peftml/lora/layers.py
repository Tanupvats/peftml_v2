"""Core LoRA layer.

Implements the Low-Rank Adaptation from Hu et al., 2022.  The trainable
delta is decomposed as ``ΔW = B · A · (α / r)`` where ``A ∈ R^{r×d_in}``
and ``B ∈ R^{d_out×r}``.  ``B`` is initialised to zero so the model's
output is unchanged at the start of training.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.utils import get_logger

logger = get_logger(__name__)

__all__ = ["LoRALinear"]


class LoRALinear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` with a frozen base and trainable LoRA adapters.

    Parameters
    ----------
    base_layer : nn.Linear
        The original (frozen) linear layer.
    r : int
        LoRA rank.
    lora_alpha : int
        Scaling numerator (``scaling = alpha / r``).
    lora_dropout : float
        Dropout applied to the input before the low-rank path.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self._merged = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # Freeze base
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        # Adapter matrices
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Kaiming on A, zeros on B → initial ΔW = 0."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def merged(self) -> bool:
        return self._merged

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)

        if self._merged:
            return base_out

        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + lora_out * self.scaling

    # -----------------------------------------------------------------
    # Merge / Unmerge for deployment
    # -----------------------------------------------------------------
    def merge(self) -> None:
        """Bake LoRA weights into the base layer for zero-overhead inference."""
        if self._merged:
            return
        with torch.no_grad():
            self.base_layer.weight.data += (self.lora_B @ self.lora_A) * self.scaling
        self._merged = True
        logger.debug("Merged LoRA adapter.")

    def unmerge(self) -> None:
        """Reverse a previous :meth:`merge`."""
        if not self._merged:
            return
        with torch.no_grad():
            self.base_layer.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
        self._merged = False
        logger.debug("Unmerged LoRA adapter.")

    def extra_repr(self) -> str:
        return (
            f"in={self.base_layer.in_features}, out={self.base_layer.out_features}, "
            f"r={self.r}, alpha={self.lora_alpha}, merged={self._merged}"
        )
