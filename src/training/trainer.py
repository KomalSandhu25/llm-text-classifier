"""
HuggingFace Trainer subclass with automatic MLflow experiment tracking.

Every training run is logged as an MLflow run under the experiment name
defined in :class:`~src.config.Settings`.  The following are recorded
automatically without any changes to training scripts:

* All hyperparameters from :class:`~src.config.Settings`
* Train and eval metrics at every logging step
* Best model checkpoint as an MLflow artifact
* Confusion matrix PNG after the final evaluation epoch

Usage::

    from src.config import Settings
    from src.training.trainer import MLflowTrainer

    cfg = Settings()
    trainer = MLflowTrainer.from_config(
        cfg=cfg,
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    EvalPrediction,
    PreTrainedModel,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from src.config import Settings

logger = logging.getLogger(__name__)


class MLflowTrainer(Trainer):
    """HuggingFace Trainer with automatic MLflow integration.

    Starts an MLflow run on :meth:`train`, logs all config hyperparameters,
    records metrics at every eval step, saves the best checkpoint as a
    logged artifact, and optionally logs a confusion-matrix PNG at the end.

    Args:
        cfg: Application settings; all fields are logged as MLflow params.
        mlflow_experiment: MLflow experiment name.  Defaults to
            ``cfg.mlflow_experiment_name``.
        log_confusion_matrix: If ``True``, generate a per-class confusion
            matrix PNG and log it as an MLflow artifact after training.
        *args: Positional args forwarded to :class:`~transformers.Trainer`.
        **kwargs: Keyword args forwarded to :class:`~transformers.Trainer`.

    Example::

        trainer = MLflowTrainer(
            cfg=Settings(),
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            compute_metrics=compute_metrics,
        )
        trainer.train()
    """

    def __init__(
        self,
        cfg: Settings,
        *args: Any,
        mlflow_experiment: Optional[str] = None,
        log_confusion_matrix: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.mlflow_experiment = mlflow_experiment or cfg.mlflow_experiment_name
        self.log_confusion_matrix = log_confusion_matrix
        self._mlflow_run: Optional[mlflow.ActiveRun] = None

    # ------------------------------------------------------------------
    # Class constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        cfg: Settings,
        model: PreTrainedModel,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        output_dir: Optional[str] = None,
    ) -> "MLflowTrainer":
        """Build a fully-configured :class:`MLflowTrainer` from settings.

        Args:
            cfg: Application settings.
            model: Model to train.
            train_dataset: Training split.
            eval_dataset: Validation split.
            compute_metrics: Optional metrics callback for
                :class:`~transformers.Trainer`.
            output_dir: Override for checkpoint output directory.

        Returns:
            Ready-to-use :class:`MLflowTrainer` instance.
        """
        training_args = TrainingArguments(
            output_dir=output_dir or cfg.checkpoint_dir,
            num_train_epochs=cfg.num_epochs,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            warmup_ratio=cfg.warmup_ratio,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_micro",
            greater_is_better=True,
            logging_dir=str(Path(cfg.checkpoint_dir) / "logs"),
            logging_steps=50,
            report_to=[],          # disable default MLflow / WandB — we handle it
            fp16=cfg.fp16 and torch.cuda.is_available(),
        )
        return cls(
            cfg=cfg,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )

    # ------------------------------------------------------------------
    # MLflow helpers
    # ------------------------------------------------------------------

    def _log_config_params(self) -> None:
        """Log all Settings fields as MLflow parameters."""
        for field_name, value in self.cfg.model_dump().items():
            try:
                mlflow.log_param(field_name, value)
            except Exception:
                pass  # silently skip non-serialisable fields

    def _log_training_args(self) -> None:
        """Log key TrainingArguments fields as MLflow parameters."""
        args = self.args
        mlflow.log_params({
            "num_train_epochs": args.num_train_epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "fp16": args.fp16,
        })

    def _log_confusion_matrix_artifact(self) -> None:
        """Generate and log a per-class confusion matrix as a PNG artifact."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.metrics import multilabel_confusion_matrix

            if self.eval_dataset is None:
                return

            preds_output = self.predict(self.eval_dataset)
            logits = preds_output.predictions
            labels = preds_output.label_ids.astype(int)
            preds  = (1 / (1 + np.exp(-logits)) >= 0.5).astype(int)

            cm = multilabel_confusion_matrix(labels, preds)
            n_classes = cm.shape[0]
            fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 4))
            if n_classes == 1:
                axes = [axes]

            for i, (ax, matrix) in enumerate(zip(axes, cm)):
                im = ax.imshow(matrix, cmap="Blues")
                ax.set_title(f"Class {i}")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                fig.colorbar(im, ax=ax)

            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            mlflow.log_figure(fig, "confusion_matrix.png")
            plt.close(fig)
            logger.info("Logged confusion matrix artifact to MLflow")
        except Exception as exc:
            logger.warning("Could not log confusion matrix: %s", exc)

    # ------------------------------------------------------------------
    # Trainer overrides
    # ------------------------------------------------------------------

    def train(self, *args: Any, **kwargs: Any) -> Any:
        """Run training with automatic MLflow tracking.

        Starts an MLflow run, logs all hyperparameters, trains the model,
        logs the best checkpoint as an artifact, and (optionally) logs a
        confusion matrix PNG.

        Returns:
            :class:`~transformers.trainer_utils.TrainOutput` from the
            underlying :class:`~transformers.Trainer`.
        """
        mlflow.set_tracking_uri(self.cfg.mlflow_tracking_uri)
        mlflow.set_experiment(self.mlflow_experiment)

        with mlflow.start_run() as run:
            self._mlflow_run = run
            logger.info("MLflow run started: %s", run.info.run_id)

            self._log_config_params()
            self._log_training_args()

            result = super().train(*args, **kwargs)

            # Log final metrics
            for key, value in result.metrics.items():
                mlflow.log_metric(key, value)

            # Log best checkpoint as artifact
            best_ckpt = getattr(self.state, "best_model_checkpoint", None)
            if best_ckpt and Path(best_ckpt).exists():
                mlflow.log_artifacts(best_ckpt, artifact_path="best_checkpoint")
                logger.info("Logged best checkpoint: %s", best_ckpt)

            if self.log_confusion_matrix:
                self._log_confusion_matrix_artifact()

        return result

    def log(self, logs: dict[str, float]) -> None:
        """Override Trainer.log to mirror metrics to MLflow in real time.

        Args:
            logs: Dict of metric name → value produced by Trainer.
        """
        super().log(logs)
        step = self.state.global_step if self.state else 0
        if mlflow.active_run():
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value, step=step)
