"""Dataclass-based configuration for all peftml components."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    DETECTION = "detection"
    LANGUAGE_MODELING = "language_modeling"


class PruningMethod(str, Enum):
    GLOBAL = "global"
    UNSTRUCTURED = "unstructured"
    STRUCTURED = "structured"


class QuantMethod(str, Enum):
    LSQ = "lsq"
    PACT = "pact"
    SMOOTHQUANT = "smoothquant"


# ---------------------------------------------------------------------------
# Quantization configs
# ---------------------------------------------------------------------------
@dataclass
class LSQConfig:
    bits: int = 8
    symmetric: bool = True
    per_channel: bool = False


@dataclass
class PACTConfig:
    bits: int = 8
    alpha_init: float = 6.0


@dataclass
class SmoothQuantConfig:
    alpha: float = 0.5
    calibration_batches: int = 8
    device: str = "cuda"


# ---------------------------------------------------------------------------
# Pruning configs
# ---------------------------------------------------------------------------
@dataclass
class PruningConfig:
    method: PruningMethod = PruningMethod.GLOBAL
    amount: float = 0.3
    ignore_layers: List[str] = field(default_factory=list)


@dataclass
class IterativePruningConfig:
    target_sparsity: float = 0.5
    total_steps: int = 30
    initial_warmup_steps: int = 0
    exponent: float = 3.0


# ---------------------------------------------------------------------------
# Knowledge distillation configs
# ---------------------------------------------------------------------------
@dataclass
class FeatureMapping:
    """Describes a student→teacher intermediate feature pairing."""
    student_layer: str
    teacher_layer: str
    student_channels: int
    teacher_channels: int


@dataclass
class KDConfig:
    task_type: TaskType = TaskType.CLASSIFICATION
    temperature: float = 4.0
    alpha: float = 0.5
    beta: float = 50.0
    feature_mappings: List[FeatureMapping] = field(default_factory=list)


# ---------------------------------------------------------------------------
# LoRA / QLoRA configs
# ---------------------------------------------------------------------------
@dataclass
class LoRAConfig:
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    merge_weights: bool = True
    fan_in_fan_out: bool = False


@dataclass
class QLoRAConfig(LoRAConfig):
    """Extends LoRA with QLoRA-specific mixed-precision rules."""
    compute_dtype: str = "bfloat16"
    quant_type: str = "nf4"
    double_quant: bool = True
    train_norm_layers: bool = True


# ---------------------------------------------------------------------------
# Pipeline configs
# ---------------------------------------------------------------------------
@dataclass
class SparseQATConfig:
    task_type: TaskType = TaskType.CLASSIFICATION
    bits: int = 8
    target_sparsity: float = 0.5
    pruning_steps: int = 30
    ignore_layers: List[str] = field(default_factory=list)
