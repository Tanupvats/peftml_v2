"""Shared utility functions used across peftml."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple, Type, Union

import torch
import torch.nn as nn


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Model inspection helpers
# ---------------------------------------------------------------------------
def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Return total, trainable, and frozen parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_pct": (trainable / total * 100) if total > 0 else 0.0,
    }


def get_modules_by_type(
    model: nn.Module,
    module_types: Union[Type[nn.Module], Tuple[Type[nn.Module], ...]],
    ignore_names: Optional[List[str]] = None,
) -> List[Tuple[str, nn.Module]]:
    """Collect named modules matching *module_types*, optionally filtering by name."""
    ignore_names = ignore_names or []
    results = []
    for name, module in model.named_modules():
        if isinstance(module, module_types):
            if not any(ign in name for ign in ignore_names):
                results.append((name, module))
    return results


def freeze(model: nn.Module) -> nn.Module:
    """Freeze all parameters of *model* in-place."""
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def unfreeze(model: nn.Module) -> nn.Module:
    """Unfreeze all parameters of *model* in-place."""
    for p in model.parameters():
        p.requires_grad = True
    return model


def compute_model_sparsity(model: nn.Module) -> Dict[str, float]:
    """Compute the overall weight sparsity (fraction of exact zeros)."""
    total_zeros = 0
    total_elements = 0
    layer_sparsity: Dict[str, float] = {}

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight
            zeros = float((w == 0).sum())
            elements = float(w.nelement())
            total_zeros += zeros
            total_elements += elements
            if elements > 0:
                layer_sparsity[name] = zeros / elements * 100

    global_pct = (total_zeros / total_elements * 100) if total_elements > 0 else 0.0
    return {"global_sparsity_pct": global_pct, "layer_sparsity_pct": layer_sparsity}


def set_attr_by_name(model: nn.Module, target: str, value: nn.Module) -> None:
    """Programmatically set a submodule by its dotted name.

    ``target='layer1.0.conv1'`` → ``model.layer1[0].conv1 = value``
    """
    atoms = target.split(".")
    parent = model
    for atom in atoms[:-1]:
        parent = getattr(parent, atom) if not atom.isdigit() else parent[int(atom)]
    setattr(parent, atoms[-1], value)


def get_attr_by_name(model: nn.Module, target: str) -> nn.Module:
    """Programmatically get a submodule by its dotted name."""
    atoms = target.split(".")
    current = model
    for atom in atoms:
        current = getattr(current, atom) if not atom.isdigit() else current[int(atom)]
    return current
