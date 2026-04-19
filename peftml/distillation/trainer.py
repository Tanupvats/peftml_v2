"""Dynamic Knowledge Distillation trainer.

Wraps a student model and a frozen teacher model in a single ``nn.Module``
that computes the combined task + logit KD + intermediate feature KD loss
in one ``.forward()`` call.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.config import FeatureMapping, KDConfig, TaskType
from ..core.utils import freeze, get_logger
from .adapters import ChannelAdapter
from .losses import hinton_kd_loss, spatial_feature_loss

logger = get_logger(__name__)

__all__ = ["DynamicKDTrainer"]


class DynamicKDTrainer(nn.Module):
    """Unified knowledge-distillation training wrapper.

    Parameters
    ----------
    student : nn.Module
        The model to be trained.
    teacher : nn.Module
        A pre-trained model that will be frozen.
    config : KDConfig
        Full distillation configuration (task type, temperature, loss
        weights, feature mappings).
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        config: Optional[KDConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config or KDConfig()

        self.student = student
        self.teacher = freeze(teacher)

        # Feature-distillation infrastructure
        self.adaptors = nn.ModuleDict()
        self._s_acts: Dict[str, torch.Tensor] = {}
        self._t_acts: Dict[str, torch.Tensor] = {}
        self._hooks: list = []

        if self.config.feature_mappings:
            self._setup_feature_hooks()

    # -----------------------------------------------------------------
    # Hook management
    # -----------------------------------------------------------------
    def _setup_feature_hooks(self) -> None:
        def _make_hook(name: str, storage: dict):
            def hook(_m, _i, output):
                storage[name] = output
            return hook

        modules_s = dict(self.student.named_modules())
        modules_t = dict(self.teacher.named_modules())

        for fm in self.config.feature_mappings:
            # Channel adapter (only if dimensions differ)
            if fm.student_channels != fm.teacher_channels:
                self.adaptors[fm.student_layer] = ChannelAdapter(fm.student_channels, fm.teacher_channels)

            s_mod = modules_s.get(fm.student_layer)
            t_mod = modules_t.get(fm.teacher_layer)
            if s_mod is None:
                raise ValueError(f"Student layer '{fm.student_layer}' not found.")
            if t_mod is None:
                raise ValueError(f"Teacher layer '{fm.teacher_layer}' not found.")

            self._hooks.append(s_mod.register_forward_hook(_make_hook(fm.student_layer, self._s_acts)))
            self._hooks.append(t_mod.register_forward_hook(_make_hook(fm.teacher_layer, self._t_acts)))

        logger.info("Set up %d feature-distillation hooks.", len(self.config.feature_mappings))

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------
    def forward(
        self,
        inputs: Any,
        targets: Any,
        criterion: nn.Module,
    ) -> Tuple[torch.Tensor, Any]:
        """Compute combined loss.

        Returns ``(total_loss, student_outputs)`` so the caller can log
        accuracy / mIoU on the student predictions.
        """
        # Student forward
        student_out = self.student(inputs)

        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_out = self.teacher(inputs)

        total_loss = self._task_and_logit_loss(student_out, teacher_out, targets, criterion)

        # Feature distillation
        if self.config.feature_mappings:
            total_loss = total_loss + self.config.beta * self._feature_loss()

        # Housekeeping
        self._s_acts.clear()
        self._t_acts.clear()

        return total_loss, student_out

    # -----------------------------------------------------------------
    # Loss helpers
    # -----------------------------------------------------------------
    def _task_and_logit_loss(self, s_out, t_out, targets, criterion) -> torch.Tensor:
        cfg = self.config

        if cfg.task_type == TaskType.CLASSIFICATION:
            task = criterion(s_out, targets)
            kd = hinton_kd_loss(s_out, t_out, cfg.temperature)
            return cfg.alpha * task + (1.0 - cfg.alpha) * kd

        if cfg.task_type == TaskType.SEGMENTATION:
            s_map = s_out["out"] if isinstance(s_out, dict) else s_out
            t_map = t_out["out"] if isinstance(t_out, dict) else t_out
            task = criterion(s_map, targets)
            kd = spatial_feature_loss(s_map, t_map)
            return task + cfg.alpha * kd

        if cfg.task_type == TaskType.DETECTION:
            # Detection models in torchvision return a loss dict during training
            loss_dict = self.student(inputs=None, targets=targets)  # already forwarded
            return sum(loss_dict.values())

        if cfg.task_type == TaskType.LANGUAGE_MODELING:
            # For LLMs: cross-entropy on tokens + KD on logit distributions
            task = criterion(s_out.view(-1, s_out.size(-1)), targets.view(-1))
            kd = hinton_kd_loss(
                s_out.view(-1, s_out.size(-1)),
                t_out.view(-1, t_out.size(-1)),
                cfg.temperature,
            )
            return cfg.alpha * task + (1.0 - cfg.alpha) * kd

        raise ValueError(f"Unknown task_type: {cfg.task_type}")

    def _feature_loss(self) -> torch.Tensor:
        loss = torch.tensor(0.0, device=next(self.student.parameters()).device)
        for fm in self.config.feature_mappings:
            s_feat = self._s_acts[fm.student_layer]
            t_feat = self._t_acts[fm.teacher_layer]

            if fm.student_layer in self.adaptors:
                s_feat = self.adaptors[fm.student_layer](s_feat)

            # Spatial alignment when stride differs
            if s_feat.shape[2:] != t_feat.shape[2:]:
                s_feat = F.interpolate(s_feat, size=t_feat.shape[2:], mode="bilinear", align_corners=False)

            loss = loss + spatial_feature_loss(s_feat, t_feat)
        return loss

    # -----------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------
    def teardown(self) -> None:
        """Remove forward hooks to prevent memory leaks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        logger.info("Removed KD feature hooks.")
