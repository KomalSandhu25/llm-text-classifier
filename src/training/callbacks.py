"""
Custom HuggingFace training callbacks.

Provides:

* :class:`EarlyStoppingCallback` — stops training when a monitored metric
  stops improving, preventing overfitting on small datasets.
* :class:`MLflowLoggingCallback` — logs epoch-level summaries to MLflow,
  complementing the step-level logging done inside :class:`~src.training.trainer.MLflowTrainer`.

Usage::

    from transformers import Trainer
    from src.training.callbacks import EarlyStoppingCallback, MLflowLoggingCallback

    trainer = Trainer(
        ...,
        callbacks=[
            EarlyStoppingCallback(patience=3, metric="eval_f1_micro"),
            MLflowLoggingCallback(),
        ],
    )
"""

from __future__ import annotations

import logging
from typing import Optional

import mlflow
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)


class EarlyStoppingCallback(TrainerCallback):
    """Stop training when a monitored eval metric stops improving.

    After each evaluation, the callback checks whether the monitored metric
    has improved by at least ``min_delta`` relative to the best seen value.
    If it has not improved for ``patience`` consecutive evaluations, training
    is stopped by setting ``control.should_training_stop = True``.

    Args:
        patience: Number of evaluations with no improvement before stopping.
        metric: Name of the eval metric to monitor, e.g. ``"eval_f1_micro"``.
        min_delta: Minimum change to count as an improvement.
        greater_is_better: Set to ``False`` for loss-based metrics.

    Example::

        cb = EarlyStoppingCallback(patience=3, metric="eval_f1_micro")
        trainer = Trainer(..., callbacks=[cb])
    """

    def __init__(
        self,
        patience: int = 3,
        metric: str = "eval_f1_micro",
        min_delta: float = 1e-4,
        greater_is_better: bool = True,
    ) -> None:
        self.patience = patience
        self.metric = metric
        self.min_delta = min_delta
        self.greater_is_better = greater_is_better

        self._best_value: Optional[float] = None
        self._no_improve_count: int = 0

    def _is_improvement(self, current: float) -> bool:
        """Return True if ``current`` is a meaningful improvement over best.

        Args:
            current: Latest metric value.

        Returns:
            ``True`` if the metric improved by at least ``min_delta``.
        """
        if self._best_value is None:
            return True
        if self.greater_is_better:
            return current > self._best_value + self.min_delta
        return current < self._best_value - self.min_delta

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[dict[str, float]] = None,
        **kwargs: object,
    ) -> TrainerControl:
        """Check metric after each evaluation and trigger early stopping if needed.

        Args:
            args: Training arguments (unused but required by callback API).
            state: Current trainer state.
            control: Trainer control object; sets ``should_training_stop``.
            metrics: Evaluation metrics dict produced by Trainer.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Modified :class:`~transformers.TrainerControl`.
        """
        if metrics is None:
            return control

        current = metrics.get(self.metric)
        if current is None:
            logger.warning(
                "EarlyStoppingCallback: metric '%s' not found in eval metrics %s",
                self.metric, list(metrics.keys()),
            )
            return control

        if self._is_improvement(current):
            self._best_value = current
            self._no_improve_count = 0
            logger.info("EarlyStopping: %s improved to %.5f", self.metric, current)
        else:
            self._no_improve_count += 1
            logger.info(
                "EarlyStopping: no improvement (%d/%d) — best=%.5f, current=%.5f",
                self._no_improve_count, self.patience, self._best_value, current,
            )

        if self._no_improve_count >= self.patience:
            logger.info(
                "EarlyStopping: patience %d reached, stopping training at epoch %.1f",
                self.patience, state.epoch or 0,
            )
            control.should_training_stop = True

        return control


class MLflowLoggingCallback(TrainerCallback):
    """Log epoch-level metric summaries to MLflow after each evaluation.

    Works alongside the step-level logging in :class:`~src.training.trainer.MLflowTrainer`.
    Useful when the trainer is used without subclassing (e.g. vanilla
    :class:`~transformers.Trainer`).

    Example::

        trainer = Trainer(..., callbacks=[MLflowLoggingCallback()])
    """

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[dict[str, float]] = None,
        **kwargs: object,
    ) -> None:
        """Log eval metrics to the active MLflow run (if any).

        Args:
            args: Training arguments (unused).
            state: Current trainer state; provides ``global_step``.
            control: Trainer control object (not modified).
            metrics: Evaluation metrics from Trainer.
            **kwargs: Additional keyword arguments (ignored).
        """
        if not mlflow.active_run() or metrics is None:
            return

        step = state.global_step
        epoch = int(state.epoch) if state.epoch else 0
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value, step=step)

        logger.debug("MLflowLoggingCallback: logged %d metrics at step %d (epoch %d)",
                     len(metrics), step, epoch)
