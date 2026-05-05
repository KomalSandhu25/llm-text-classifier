"""
Evaluation metrics for multi-label text classification.

Provides :func:`compute_metrics`, a callback-compatible function for the
HuggingFace :class:`~transformers.Trainer` that converts raw logits into
threshold-binarised predictions and computes a full suite of multi-label
classification metrics using ``scikit-learn``.

Metrics computed
----------------
* **Micro F1** — treats every (sample, label) pair as a single binary
  prediction.  Dominated by frequent classes.
* **Macro F1** — unweighted mean of per-class F1.  Each class contributes
  equally regardless of support.
* **Weighted F1** — mean of per-class F1 weighted by class support.
* **Exact-match accuracy** (subset accuracy) — fraction of samples where the
  full predicted label set equals the true label set.
* **Per-class precision, recall, F1** — returned as nested dicts so they can
  be logged individually to MLflow / W&B.

Usage::

    from transformers import Trainer, TrainingArguments
    from src.model.metrics import compute_metrics

    trainer = Trainer(
        model=model,
        args=TrainingArguments(...),
        compute_metrics=compute_metrics,
    )
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import EvalPrediction


# Default sigmoid threshold for converting logits → binary predictions.
_DEFAULT_THRESHOLD: float = 0.5


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid.

    Args:
        x: Input array of arbitrary shape.

    Returns:
        Array of same shape with values in ``(0, 1)``.
    """
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def binarise_logits(
    logits: np.ndarray,
    threshold: float = _DEFAULT_THRESHOLD,
) -> np.ndarray:
    """Apply sigmoid then threshold logits to produce binary predictions.

    Args:
        logits: Raw model output, shape ``(N, num_labels)``.
        threshold: Decision boundary; predictions above this value are set to 1.

    Returns:
        Integer array of shape ``(N, num_labels)`` with values in ``{0, 1}``.

    Example::

        preds = binarise_logits(logits, threshold=0.5)
    """
    probs = _sigmoid(logits)
    return (probs >= threshold).astype(int)


def per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute per-class precision, recall, and F1.

    Args:
        y_true: Ground-truth binary matrix, shape ``(N, C)``.
        y_pred: Predicted binary matrix, shape ``(N, C)``.
        label_names: Optional list of ``C`` class names for readable keys.
            Falls back to ``"class_0"``, ``"class_1"``, … if not provided.

    Returns:
        Dict mapping class name → ``{"precision": float, "recall": float,
        "f1": float, "support": int}``.

    Example::

        metrics = per_class_metrics(y_true, y_pred, ["World", "Sports", "Business", "Tech"])
    """
    num_classes = y_true.shape[1]
    names = label_names or [f"class_{i}" for i in range(num_classes)]

    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    support = y_true.sum(axis=0).astype(int)

    return {
        name: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(names)
    }


def compute_metrics(
    eval_pred: EvalPrediction,
    threshold: float = _DEFAULT_THRESHOLD,
    label_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute multi-label classification metrics from a HuggingFace EvalPrediction.

    This function is designed to be passed directly to
    :class:`~transformers.Trainer` as the ``compute_metrics`` argument::

        trainer = Trainer(..., compute_metrics=compute_metrics)

    If you need a custom threshold or class names, wrap it in a lambda::

        trainer = Trainer(
            ...,
            compute_metrics=lambda p: compute_metrics(p, threshold=0.4,
                                                       label_names=["A", "B"]),
        )

    Args:
        eval_pred: Named tuple produced by HuggingFace Trainer containing
            ``predictions`` (logits, shape ``(N, C)``) and ``label_ids``
            (binary ground truth, shape ``(N, C)``).
        threshold: Sigmoid threshold for binarisation.  Default ``0.5``.
        label_names: Optional list of human-readable class names.

    Returns:
        Flat dict suitable for logging::

            {
                "f1_micro":    0.87,
                "f1_macro":    0.84,
                "f1_weighted": 0.86,
                "accuracy":    0.79,
                # per-class (flattened):
                "precision_World":  0.91,
                "recall_World":     0.88,
                "f1_World":         0.89,
                ...
            }

    Raises:
        ValueError: If ``eval_pred.predictions`` has unexpected shape.
    """
    logits: np.ndarray = eval_pred.predictions
    y_true: np.ndarray = eval_pred.label_ids.astype(int)

    if logits.ndim != 2:
        raise ValueError(
            f"Expected logits of shape (N, num_labels), got shape {logits.shape}"
        )

    y_pred = binarise_logits(logits, threshold=threshold)

    metrics: dict[str, Any] = {
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

    # Flatten per-class metrics into the top-level dict for Trainer compatibility
    class_metrics = per_class_metrics(y_true, y_pred, label_names=label_names)
    for class_name, class_dict in class_metrics.items():
        for metric_name, value in class_dict.items():
            metrics[f"{metric_name}_{class_name}"] = value

    return metrics


def format_metrics_report(
    metrics: dict[str, Any],
    label_names: list[str] | None = None,
) -> str:
    """Format a metrics dict into a human-readable report string.

    Args:
        metrics: Dict as returned by :func:`compute_metrics`.
        label_names: Class names, used to group per-class rows.

    Returns:
        Multi-line string suitable for printing to console or logging.

    Example::

        report = format_metrics_report(metrics, label_names=["World", "Sports"])
        print(report)
    """
    lines = [
        "=" * 50,
        "  Evaluation Results",
        "=" * 50,
        f"  F1  (micro)   : {metrics.get('f1_micro', 0):.4f}",
        f"  F1  (macro)   : {metrics.get('f1_macro', 0):.4f}",
        f"  F1  (weighted): {metrics.get('f1_weighted', 0):.4f}",
        f"  Exact match   : {metrics.get('accuracy', 0):.4f}",
        "",
        "  Per-class breakdown:",
        f"  {'Class':<16} {'P':>7} {'R':>7} {'F1':>7} {'Support':>9}",
        "  " + "-" * 44,
    ]

    num_classes = len(label_names) if label_names else 0
    names = label_names or [
        k.replace("f1_", "")
        for k in metrics
        if k.startswith("f1_") and k not in {"f1_micro", "f1_macro", "f1_weighted"}
    ]

    for name in names:
        p = metrics.get(f"precision_{name}", 0.0)
        r = metrics.get(f"recall_{name}", 0.0)
        f = metrics.get(f"f1_{name}", 0.0)
        s = metrics.get(f"support_{name}", 0)
        lines.append(f"  {name:<16} {p:>7.4f} {r:>7.4f} {f:>7.4f} {s:>9}")

    lines.append("=" * 50)
    return "\n".join(lines)
