

__version__ = "2.0.0"

# ── Core ──────────────────────────────────────────────────────────────
from .core.config import (
    FeatureMapping,
    IterativePruningConfig,
    KDConfig,
    LoRAConfig,
    LSQConfig,
    PACTConfig,
    PruningConfig,
    PruningMethod,
    QLoRAConfig,
    QuantMethod,
    SmoothQuantConfig,
    SparseQATConfig,
    TaskType,
)
from .core.utils import compute_model_sparsity, count_parameters, freeze, unfreeze

# ── Quantization ──────────────────────────────────────────────────────
from .quantization import (
    ActivationObserver,
    PACTReLU,
    QConv2d,
    QLinear,
    apply_smoothquant,
    replace_with_lsq,
    replace_with_pact,
)

# ── Pruning ───────────────────────────────────────────────────────────
from .pruning import DynamicPruner, IterativePruningScheduler

# ── Distillation ──────────────────────────────────────────────────────
from .distillation import (
    ChannelAdapter,
    DynamicKDTrainer,
    LinearAdapter,
    attention_transfer_loss,
    cosine_feature_loss,
    hinton_kd_loss,
    spatial_feature_loss,
)

# ── LoRA / QLoRA ──────────────────────────────────────────────────────
from .lora import LoRALinear, QLoRAOrchestrator

# ── Pipelines (top-level convenience) ─────────────────────────────────
from .pipelines import ModelCompressor, SparseQATPipeline

# ── Export ────────────────────────────────────────────────────────────
from .export import export_onnx

__all__ = [
    # Facade
    "ModelCompressor",
    # Configs
    "FeatureMapping",
    "IterativePruningConfig",
    "KDConfig",
    "LoRAConfig",
    "LSQConfig",
    "PACTConfig",
    "PruningConfig",
    "PruningMethod",
    "QLoRAConfig",
    "QuantMethod",
    "SmoothQuantConfig",
    "SparseQATConfig",
    "TaskType",
    # Quantization
    "ActivationObserver",
    "PACTReLU",
    "QConv2d",
    "QLinear",
    "apply_smoothquant",
    "replace_with_lsq",
    "replace_with_pact",
    # Pruning
    "DynamicPruner",
    "IterativePruningScheduler",
    # Distillation
    "ChannelAdapter",
    "DynamicKDTrainer",
    "LinearAdapter",
    "attention_transfer_loss",
    "cosine_feature_loss",
    "hinton_kd_loss",
    "spatial_feature_loss",
    # LoRA
    "LoRALinear",
    "QLoRAOrchestrator",
    # Pipelines
    "SparseQATPipeline",
    # Export
    "export_onnx",
    # Utilities
    "compute_model_sparsity",
    "count_parameters",
    "freeze",
    "unfreeze",
]
