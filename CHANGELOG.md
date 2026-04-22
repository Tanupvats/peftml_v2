# Changelog

## 2.0.0 

### Breaking changes

- Package restructured into subpackages (`peftml.quantization`, `peftml.pruning`, etc.).
  Old flat imports like `from peftml import replace_with_lsq` still work via the top-level `__init__`.
- `setup.py` replaced with `pyproject.toml` (PEP 621 / hatch backend).
- All configuration moved to dataclasses (`LoRAConfig`, `KDConfig`, `SparseQATConfig`, …).
  Old positional-arg constructors are no longer supported.
- `ModelCompressor.setup_sparse_qat()` renamed to `ModelCompressor.sparse_qat()`.
- `ModelCompressor.setup_distillation()` renamed to `ModelCompressor.distill()`.
- `SparseQATPipeline.export_ready()` renamed to `SparseQATPipeline.export()`.

### Bug fixes

- **ActivationObserver** — fixed crash on `nn.Linear` layers.  The observer previously
  called `amax(dim=(0, 2, 3))` unconditionally, which fails on 2-D tensors.  Now handles
  2-D (Linear), 3-D (transformer sequence), and 4-D (Conv) shapes.
- **`replace_with_lsq`** — now quantises both `nn.Conv2d` *and* `nn.Linear` layers.
  v1 only handled Conv2d.
- **SmoothQuant** — added absorption of `1/scale` into the preceding LayerNorm for
  mathematical equivalence.  v1 only scaled the weights, breaking the forward pass.
- **`sparse_qat.py`** — fixed relative imports (was using bare `from quantization import …`).
- **KD trainer** — now validates that student/teacher layer names exist before attaching
  hooks.  v1 silently produced `KeyError` during the forward pass.

### New features

- **`QLinear`** — LSQ-quantised `nn.Linear` for transformer weight quantization.
- **`LinearAdapter`** — projection adapter for transformer-to-transformer distillation.
- **`cosine_feature_loss`** and **`attention_transfer_loss`** — additional KD losses.
- **`TaskType.LANGUAGE_MODELING`** — KD trainer now supports LLM distillation.
- **`ActivationObserver.calibrate()`** — single-call calibration over a dataloader.
- **`QLoRAOrchestrator.get_adapter_state_dict()` / `load_adapter_state_dict()`** —
  lightweight adapter-only checkpointing.
- **`QLoRAOrchestrator.merge_adapters()` / `unmerge_adapters()`** — deployment helpers.
- **`DynamicPruner.is_pruned()`** — introspection helper.
- **`IterativePruningScheduler`** — added warmup steps and configurable polynomial exponent.
- **`export_onnx()`** — ONNX export with optional simplification.
- **Registry pattern** — extensible component lookup for future plug-in methods.
- **`count_parameters()`, `freeze()`, `unfreeze()`, `compute_model_sparsity()`** — model
  inspection utilities.
- **Comprehensive test suite** — 40+ unit and integration tests.

## 1.0.0 — Initial release

Single-file prototype covering LSQ, PACT, SmoothQuant, pruning, KD, and LoRA.
