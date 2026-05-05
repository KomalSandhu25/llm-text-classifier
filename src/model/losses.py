"""
Loss functions for multi-label text classification.

Two variants are provided:

* :class:`MultiLabelLoss` — standard ``BCEWithLogitsLoss`` suitable when the
  class distribution is roughly balanced.
* :class:`WeightedMultiLabelLoss` — per-class positive weight scaling that
  penalises the minority class more heavily; recommended for imbalanced corpora.

Both classes expose the same ``forward(logits, labels)`` signature so they can
be swapped transparently in the training loop.

Usage::

    from src.model.losses import WeightedMultiLabelLoss
    import torch

    pos_weight = torch.tensor([2.5, 1.0, 3.1, 1.8])  # one weight per class
    criterion = WeightedMultiLabelLoss(pos_weight=pos_weight)

    logits = model(input_ids, attention_mask).logits   # (B, 4)
    labels = batch["labels"].float()                   # (B, 4)
    loss   = criterion(logits, labels)
    loss.backward()
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelLoss(nn.Module):
    """Binary cross-entropy loss for multi-label classification.

    A thin wrapper around :class:`torch.nn.BCEWithLogitsLoss` that accepts raw
    logits (no sigmoid required from the model) and float label tensors.

    Args:
        reduction: Specifies the reduction to apply: ``'mean'`` (default),
            ``'sum'``, or ``'none'``.
        label_smoothing: If > 0, applies label smoothing to the targets,
            pulling them away from hard 0/1.  Typical values: 0.05–0.10.

    Example::

        criterion = MultiLabelLoss(label_smoothing=0.05)
        loss = criterion(logits, labels)   # scalar tensor
    """

    def __init__(
        self,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'")
        if not 0.0 <= label_smoothing < 0.5:
            raise ValueError(f"label_smoothing must be in [0, 0.5), got {label_smoothing}")

        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self._bce = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute BCE loss.

        Args:
            logits: Raw model outputs, shape ``(B, num_labels)``.  Must NOT
                have sigmoid applied.
            labels: Ground-truth binary matrix, shape ``(B, num_labels)``,
                values in ``{0, 1}`` (float or long, will be cast to float).

        Returns:
            Scalar loss tensor (or per-element tensor when ``reduction='none'``).
        """
        targets = labels.float()
        if self.label_smoothing > 0.0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        return self._bce(logits, targets)


class WeightedMultiLabelLoss(nn.Module):
    """Per-class weighted BCE loss for imbalanced multi-label datasets.

    Applies a per-class ``pos_weight`` that up-scales the loss contribution of
    positive samples for minority classes.  A common heuristic for computing
    the weights is::

        pos_weight[c] = (N - count_positive[c]) / count_positive[c]

    where ``N`` is the total number of training examples.  Values are then
    optionally clipped to prevent extreme up-weighting.

    Args:
        pos_weight: 1-D tensor of shape ``(num_labels,)`` containing the
            positive class weight for each label.
        reduction: ``'mean'`` | ``'sum'`` | ``'none'``.
        label_smoothing: Smoothing coefficient in ``[0, 0.5)``.
        max_weight: Clip weights to this maximum value to avoid instability.

    Example::

        counts = torch.tensor([800.0, 200.0, 600.0, 400.0])  # pos samples per class
        total  = 2000.0
        pos_weight = torch.clamp((total - counts) / counts, max=10.0)
        criterion  = WeightedMultiLabelLoss(pos_weight=pos_weight)

        loss = criterion(logits, labels)
    """

    def __init__(
        self,
        pos_weight: torch.Tensor,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        max_weight: float = 10.0,
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'")
        if not 0.0 <= label_smoothing < 0.5:
            raise ValueError(f"label_smoothing must be in [0, 0.5), got {label_smoothing}")

        self.reduction = reduction
        self.label_smoothing = label_smoothing

        clipped_weights = torch.clamp(pos_weight, max=max_weight)
        self.register_buffer("pos_weight", clipped_weights)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute weighted BCE loss.

        Args:
            logits: Raw model outputs, shape ``(B, num_labels)``.
            labels: Ground-truth binary matrix, shape ``(B, num_labels)``.

        Returns:
            Scalar loss tensor.
        """
        targets = labels.float()
        if self.label_smoothing > 0.0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        # BCEWithLogitsLoss with pos_weight broadcasts correctly over the batch
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,  # type: ignore[arg-type]
            reduction=self.reduction,
        )
        return loss

    @classmethod
    def from_label_counts(
        cls,
        label_counts: torch.Tensor,
        total_samples: int,
        max_weight: float = 10.0,
        **kwargs: object,
    ) -> "WeightedMultiLabelLoss":
        """Convenience constructor: derive pos_weight from label frequency.

        Args:
            label_counts: 1-D tensor of shape ``(num_labels,)`` with the
                number of positive examples per class.
            total_samples: Total number of training samples.
            max_weight: Upper bound for the computed weights.
            **kwargs: Extra keyword arguments forwarded to ``__init__``.

        Returns:
            :class:`WeightedMultiLabelLoss` with automatically computed weights.

        Example::

            loss_fn = WeightedMultiLabelLoss.from_label_counts(
                label_counts=torch.tensor([800, 200, 600, 400]),
                total_samples=2000,
            )
        """
        pos_weight = (total_samples - label_counts.float()) / label_counts.float().clamp(min=1)
        return cls(pos_weight=pos_weight, max_weight=max_weight, **kwargs)
