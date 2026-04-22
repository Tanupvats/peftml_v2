

from __future__ import annotations

from typing import List, Optional, Set

import torch
import torch.nn as nn

from ..core.config import LoRAConfig, QLoRAConfig
from ..core.utils import count_parameters, get_logger
from .layers import LoRALinear

logger = get_logger(__name__)

__all__ = ["QLoRAOrchestrator"]

# Layer types that should never be replaced with LoRA
_NORM_TYPES = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)


class QLoRAOrchestrator:
    """Prepare an LLM for QLoRA (or standard LoRA) fine-tuning.

    Usage::

        orch = QLoRAOrchestrator(model, config=QLoRAConfig(r=16, target_modules=["q_proj", "v_proj"]))
        model = orch.prepare()

    Parameters
    ----------
    model : nn.Module
        The base model (e.g. Llama, Mistral, Phi).
    config : LoRAConfig | QLoRAConfig
        Adapter and mixed-precision settings.
    """

    def __init__(self, model: nn.Module, config: Optional[LoRAConfig] = None) -> None:
        self.model = model
        self.config = config or LoRAConfig()
        self._injected = False


    def prepare(self) -> nn.Module:
        """Freeze → stabilise norms → inject adapters.  Returns the model."""
        self._freeze_base()
        if isinstance(self.config, QLoRAConfig) or getattr(self.config, "train_norm_layers", False):
            self._stabilise_norms()
        self._inject_adapters()
        self._log_budget()
        self._injected = True
        return self.model

    def merge_adapters(self) -> nn.Module:
        """Merge all LoRA weights into the base layers for deployment."""
        merged = 0
        for module in self.model.modules():
            if isinstance(module, LoRALinear):
                module.merge()
                merged += 1
        logger.info("Merged %d LoRA adapters into base weights.", merged)
        return self.model

    def unmerge_adapters(self) -> nn.Module:
        """Reverse all merges (e.g. to resume training)."""
        for module in self.model.modules():
            if isinstance(module, LoRALinear):
                module.unmerge()
        return self.model

    def get_adapter_state_dict(self) -> dict:
        """Return only the LoRA adapter parameters (for lightweight checkpointing)."""
        adapter_sd = {}
        for name, param in self.model.named_parameters():
            if "lora_" in name and param.requires_grad:
                adapter_sd[name] = param.data.clone()
        logger.info("Extracted %d adapter tensors.", len(adapter_sd))
        return adapter_sd

    def load_adapter_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        """Load previously saved adapter weights."""
        model_sd = self.model.state_dict()
        loaded = 0
        for key, value in state_dict.items():
            if key in model_sd:
                model_sd[key].copy_(value)
                loaded += 1
            elif strict:
                raise KeyError(f"Adapter key '{key}' not found in model.")
        logger.info("Loaded %d adapter tensors.", loaded)


    def _freeze_base(self) -> None:
        """Freeze every parameter in the model."""
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info("Froze all base model parameters.")

    def _stabilise_norms(self) -> None:
        """Cast normalisation layers to FP32 and optionally make them trainable.

        When base weights are in 4-bit NF4, norms in low precision cause
        gradient instability.
        """
        count = 0
        train_norms = getattr(self.config, "train_norm_layers", True)
        for name, module in self.model.named_modules():
            if isinstance(module, _NORM_TYPES):
                module.float()  # cast all buffers/params to FP32
                if train_norms:
                    for p in module.parameters():
                        p.requires_grad = True
                count += 1
        logger.info("Cast %d norm layers to FP32 (trainable=%s).", count, train_norms)

    def _inject_adapters(self) -> None:
        """Recursively replace targeted ``nn.Linear`` modules with :class:`LoRALinear`."""
        cfg = self.config
        targets = set(cfg.target_modules)
        replaced: List[str] = []

        for fqn, module in list(self.model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if isinstance(module, LoRALinear):
                continue  # already wrapped
            if not self._name_matches(fqn, targets):
                continue

            lora_layer = LoRALinear(
                base_layer=module,
                r=cfg.r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
            )

            # Walk to parent and replace attribute
            self._replace_in_parent(fqn, lora_layer)
            replaced.append(fqn)

        logger.info(
            "Injected LoRA adapters (r=%d, α=%d) into %d layers: %s",
            cfg.r,
            cfg.lora_alpha,
            len(replaced),
            ", ".join(replaced[:8]) + ("..." if len(replaced) > 8 else ""),
        )

    @staticmethod
    def _name_matches(fqn: str, targets: Set[str]) -> bool:
        """Check if the fully-qualified name ends with any target module name."""
        last = fqn.rsplit(".", 1)[-1]
        return last in targets

    def _replace_in_parent(self, fqn: str, new_module: nn.Module) -> None:
        atoms = fqn.split(".")
        parent = self.model
        for atom in atoms[:-1]:
            parent = getattr(parent, atom) if not atom.isdigit() else parent[int(atom)]
        setattr(parent, atoms[-1], new_module)

    def _log_budget(self) -> None:
        stats = count_parameters(self.model)
        logger.info(
            "Trainable: %s / %s (%.4f%%)",
            f"{stats['trainable']:,}",
            f"{stats['total']:,}",
            stats["trainable_pct"],
        )
