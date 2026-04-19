# peftml

**Parameter-Efficient Fine-Tuning & Machine Learning Compression Toolkit**

Prune, quantise, distil, and fine-tune deep learning models — CNNs, transformers, and LLMs — through a single, unified API.

[![System Architecture](src/peftml_system_architecture.png)]()


---

## Features

| Technique | Use case | Key classes |
|---|---|---|
| **Pruning** (global / unstructured / structured) | Remove redundant weights from any architecture | `DynamicPruner`, `IterativePruningScheduler` |
| **Quantization-Aware Training** (LSQ + PACT) | Train INT8/INT4 models that deploy without accuracy loss | `QConv2d`, `QLinear`, `PACTReLU` |
| **Post-Training Quantization** (SmoothQuant) | Quantise LLMs without retraining | `ActivationObserver`, `apply_smoothquant` |
| **Sparse QAT** | Simultaneous pruning + quantization for edge deployment | `SparseQATPipeline` |
| **Knowledge Distillation** | Train small students from large teachers (classification, segmentation, detection, LM) | `DynamicKDTrainer` |
| **LoRA / QLoRA** | Parameter-efficient LLM fine-tuning | `LoRALinear`, `QLoRAOrchestrator` |
| **ONNX Export** | Ship compressed models to production runtimes | `export_onnx` |

---

## Installation

```bash
pip install peftml

# With dev tools
pip install peftml[dev]

# With ONNX export support
pip install peftml[onnx]
```

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.0

---

## Quick start

Every technique is accessible through the `ModelCompressor` facade:

```python
from peftml import ModelCompressor
```

### 1. Pruning a CNN

[![Pruning](src/Pruning_techniques.png)]()

```python
import torchvision.models as models

model = models.resnet50(weights="DEFAULT")
comp = ModelCompressor(model)

# Global L1 pruning — remove 40% of weights network-wide
pruner = comp.prune(method="global", amount=0.4, ignore_layers=["fc"])
print(pruner.compute_sparsity())

# Commit before saving
pruner.commit()
torch.save(model.state_dict(), "resnet50_pruned.pt")
```

### 2. Quantization-Aware Training (CNN)

[![Quantization](src/Quantization_techniques.png)]()

```python
model = models.mobilenet_v2(weights="DEFAULT")
comp = ModelCompressor(model)

# Replace Conv2d → QConv2d (LSQ) and ReLU → PACTReLU
model = comp.quantize_for_qat(bits=8)

# Train normally — quantization is differentiable
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
for batch in train_loader:
    images, labels = batch
    loss = criterion(model(images), labels)
    loss.backward()
    optimizer.step()
```

### 3. Sparse QAT (Pruning + Quantization for Edge)

```python
comp = ModelCompressor(model)
pipe = comp.sparse_qat(
    task_type="classification",
    bits=8,
    target_sparsity=0.5,
    pruning_steps=30,
)

for epoch in range(30):
    for batch in train_loader:
        loss = pipe.train_step(batch, optimizer, criterion, device="cuda")
    pipe.step_epoch()

final_model = pipe.export()
```

### 4. Knowledge Distillation

```python
teacher = models.resnet50(weights="DEFAULT")
student = models.mobilenet_v3_small(weights="DEFAULT")

comp = ModelCompressor(student)
trainer = comp.distill(
    teacher=teacher,
    task_type="classification",
    temperature=4.0,
    alpha=0.5,
)

# Training loop
for batch in train_loader:
    images, labels = batch
    loss, student_out = trainer(images, labels, criterion)
    loss.backward()
    optimizer.step()

trainer.teardown()  # clean up hooks
```

### 5. QLoRA Fine-Tuning (LLMs)

[![QLoRa](src/Lora_qlora.png)]()

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

comp = ModelCompressor(model)
model = comp.apply_qlora(
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    r=16,
    lora_alpha=32,
)

# Only ~0.1% of params are trainable
# Train with your favourite loop / HF Trainer / etc.

# For deployment: merge adapters into base weights
comp.merge_lora()
```

### 6. Post-Training Quantization (SmoothQuant)

```python
model = comp.apply_smoothquant(
    dataloader=calibration_loader,
    alpha=0.5,
    calibration_batches=8,
)
```

### 7. ONNX Export

```python
from peftml import export_onnx

export_onnx(
    model,
    dummy_input=torch.randn(1, 3, 224, 224),
    path="model.onnx",
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
)
```

---

## Advanced: Direct API Access

For fine-grained control, use the submodules directly:

```python
from peftml.quantization import replace_with_lsq, replace_with_pact
from peftml.pruning import DynamicPruner, IterativePruningScheduler
from peftml.distillation import DynamicKDTrainer, hinton_kd_loss
from peftml.lora import LoRALinear, QLoRAOrchestrator
from peftml.core import LoRAConfig, KDConfig, FeatureMapping
```

---

## Configuration

All components accept dataclass configs for reproducibility:

```python
from peftml import LoRAConfig, QLoRAConfig, KDConfig, SparseQATConfig, FeatureMapping, TaskType

lora_cfg = LoRAConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])

kd_cfg = KDConfig(
    task_type=TaskType.SEGMENTATION,
    temperature=4.0,
    alpha=0.3,
    beta=50.0,
    feature_mappings=[
        FeatureMapping("backbone.layer3", "backbone.layer3", 256, 1024),
    ],
)
```

---

## Project Structure

```
peftml/
├── core/           # Configs, registry, utilities
├── quantization/   # LSQ, PACT, SmoothQuant, observers
├── pruning/        # DynamicPruner, iterative schedulers
├── distillation/   # KD losses, adapters, trainer
├── lora/           # LoRA layers, QLoRA orchestrator
├── pipelines/      # SparseQAT, ModelCompressor facade
└── export/         # ONNX export
```

---

## Testing

```bash
pip install peftml[dev]
pytest tests/ -v
```

---

## License

Apache 2.0
