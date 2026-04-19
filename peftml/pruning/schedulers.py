"""Iterative pruning schedulers.

Gradually increasing sparsity over training avoids the accuracy collapse
seen with one-shot pruning.  The default cubic polynomial schedule follows
Zhu & Gupta, 2017 ("To Prune, or Not to Prune").
"""

from __future__ import annotations

from ..core.config import IterativePruningConfig
from ..core.utils import get_logger
from .pruner import DynamicPruner

logger = get_logger(__name__)

__all__ = ["IterativePruningScheduler"]


class IterativePruningScheduler:
    """Polynomial sparsity ramp-up.

    At each call to :meth:`step`, the scheduler commits the current mask,
    computes the target sparsity for the next interval, and re-applies
    global pruning to reach that target.

    The target at step *t* is::

        s(t) = s_f · (1 − (1 − t / T)^n)

    where ``s_f`` is *target_sparsity*, ``T`` is *total_steps*, and ``n``
    is the polynomial exponent (default 3).
    """

    def __init__(
        self,
        pruner: DynamicPruner,
        target_sparsity: float,
        total_steps: int,
        initial_warmup_steps: int = 0,
        exponent: float = 3.0,
    ) -> None:
        if not 0.0 < target_sparsity < 1.0:
            raise ValueError(f"target_sparsity must be in (0, 1), got {target_sparsity}")
        if total_steps < 1:
            raise ValueError(f"total_steps must be >= 1, got {total_steps}")

        self.pruner = pruner
        self.target_sparsity = target_sparsity
        self.total_steps = total_steps
        self.warmup = initial_warmup_steps
        self.exponent = exponent
        self._current_step = 0

    @classmethod
    def from_config(cls, pruner: DynamicPruner, config: IterativePruningConfig) -> "IterativePruningScheduler":
        return cls(
            pruner=pruner,
            target_sparsity=config.target_sparsity,
            total_steps=config.total_steps,
            initial_warmup_steps=config.initial_warmup_steps,
            exponent=config.exponent,
        )

    @property
    def current_target(self) -> float:
        """Sparsity target for the current step."""
        if self._current_step < self.warmup:
            return 0.0
        effective_step = self._current_step - self.warmup
        effective_total = self.total_steps - self.warmup
        if effective_total <= 0:
            return self.target_sparsity
        progress = min(effective_step / effective_total, 1.0)
        return self.target_sparsity * (1.0 - (1.0 - progress) ** self.exponent)

    @property
    def finished(self) -> bool:
        return self._current_step >= self.total_steps

    def step(self) -> float:
        """Advance one step: commit existing masks, compute new target, re-prune.

        Returns the sparsity target applied at this step.
        """
        if self.finished:
            return self.target_sparsity

        target = self.current_target
        self._current_step += 1

        if target <= 0.0:
            logger.debug("Warmup step %d — no pruning applied.", self._current_step)
            return 0.0

        # Commit bakes current zeros into the weight tensor so that
        # the subsequent `apply_global` treats them as regular zeros.
        # Because L1-unstructured sorts by magnitude, already-zero weights
        # are guaranteed to stay pruned, and `target` is an *absolute*
        # fraction of all weights.  This is correct.
        self.pruner.commit()
        self.pruner.apply_global(amount=target)

        stats = self.pruner.compute_sparsity()
        logger.info(
            "Pruning step %d/%d — target %.1f%% — actual %.1f%%",
            self._current_step,
            self.total_steps,
            target * 100,
            stats["global_sparsity_pct"],
        )
        return target
