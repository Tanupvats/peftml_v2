

from __future__ import annotations

import torch


class _RoundSTE(torch.autograd.Function):
    """Round with straight-through gradient (identity on backward)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output


class _GradScale(torch.autograd.Function):
    """Scale the gradient of a tensor by a fixed factor.

    Used in LSQ to prevent the learned step-size ``s`` from receiving
    oversized gradients (accumulated over the full weight tensor).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale_factor: float) -> torch.Tensor:  # noqa: D401
        ctx.scale_factor = scale_factor
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output * ctx.scale_factor, None


class _FloorSTE(torch.autograd.Function):
    """Floor with straight-through gradient."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return torch.floor(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output


# ---------------------------------------------------------------------------
# Public functional API
# ---------------------------------------------------------------------------

def round_ste(x: torch.Tensor) -> torch.Tensor:
    """Differentiable rounding (STE)."""
    return _RoundSTE.apply(x)


def floor_ste(x: torch.Tensor) -> torch.Tensor:
    """Differentiable floor (STE)."""
    return _FloorSTE.apply(x)


def grad_scale(x: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """Scale the backward gradient of *x* by *scale_factor*."""
    return _GradScale.apply(x, scale_factor)
