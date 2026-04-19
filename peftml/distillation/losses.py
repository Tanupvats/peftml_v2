"""Knowledge distillation loss functions.

Covers logit-level (Hinton KD), spatial feature (MSE), and
cosine-similarity losses for classification, segmentation, and detection.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "hinton_kd_loss",
    "spatial_feature_loss",
    "cosine_feature_loss",
    "attention_transfer_loss",
]


def hinton_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 4.0,
) -> torch.Tensor:
    """Standard Hinton KL-divergence distillation loss.

    Softens both distributions by *temperature* and scales the loss by
    ``T²`` to maintain gradient magnitudes.
    """
    return (
        F.kl_div(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1),
            reduction="batchmean",
        )
        * (temperature ** 2)
    )


def spatial_feature_loss(
    student_feats: torch.Tensor,
    teacher_feats: torch.Tensor,
) -> torch.Tensor:
    """L2-normalised MSE between intermediate feature maps.

    Normalizing first prevents the teacher's larger activations from
    dominating the loss landscape.
    """
    s = F.normalize(student_feats.flatten(2), p=2, dim=2)
    t = F.normalize(teacher_feats.flatten(2), p=2, dim=2)
    return F.mse_loss(s, t)


def cosine_feature_loss(
    student_feats: torch.Tensor,
    teacher_feats: torch.Tensor,
) -> torch.Tensor:
    """1 − cosine similarity averaged over the spatial dimensions."""
    s = student_feats.flatten(2)
    t = teacher_feats.flatten(2)
    cos = F.cosine_similarity(s, t, dim=1)
    return (1.0 - cos).mean()


def attention_transfer_loss(
    student_feats: torch.Tensor,
    teacher_feats: torch.Tensor,
    p: int = 2,
) -> torch.Tensor:
    """Attention Transfer (Zagoruyko & Komodakis, 2017).

    Matches the spatial attention maps (L_p-norm across channels) between
    student and teacher.
    """

    def _attention_map(feat: torch.Tensor) -> torch.Tensor:
        return F.normalize(feat.pow(p).mean(dim=1, keepdim=True).flatten(2), p=2, dim=2)

    return F.mse_loss(_attention_map(student_feats), _attention_map(teacher_feats))
