"""
CLI training entry point for llm-text-classifier.

Parses command-line arguments, initialises the MLflow experiment, builds the
model and data loaders, and launches :class:`~src.training.trainer.MLflowTrainer`.

Usage::

    python scripts/train.py --model distilbert-base-uncased --epochs 5 --lr 2e-5

All arguments override the corresponding ``Settings`` field; unspecified
arguments fall back to the values in ``.env`` or the class defaults.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure src/ is importable when running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Settings
from src.data.loaders import build_dataloaders
from src.model.classifier import DistilBertClassifier
from src.model.metrics import compute_metrics
from src.training.trainer import MLflowTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed :class:`argparse.Namespace` with all training options.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT for multi-label text classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default=None,
                        help="HuggingFace model name or path (overrides MODEL_NAME env)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs (overrides NUM_EPOCHS env)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (overrides LEARNING_RATE env)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Per-device batch size (overrides BATCH_SIZE env)")
    parser.add_argument("--max-seq-len", type=int, default=None,
                        help="Maximum tokenisation length (overrides MAX_SEQ_LENGTH env)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save checkpoints")
    parser.add_argument("--experiment", type=str, default=None,
                        help="MLflow experiment name")
    parser.add_argument("--freeze-base", action="store_true",
                        help="Freeze DistilBERT weights and train only the head")
    parser.add_argument("--no-confusion-matrix", action="store_true",
                        help="Skip logging confusion matrix artifact to MLflow")
    return parser.parse_args()


def main() -> None:
    """Main training entry point."""
    args = parse_args()

    # Build settings, applying CLI overrides
    cfg_overrides: dict[str, object] = {}
    if args.model:       cfg_overrides["model_name"] = args.model
    if args.epochs:      cfg_overrides["num_epochs"] = args.epochs
    if args.lr:          cfg_overrides["learning_rate"] = args.lr
    if args.batch_size:  cfg_overrides["batch_size"] = args.batch_size
    if args.max_seq_len: cfg_overrides["max_seq_length"] = args.max_seq_len
    if args.experiment:  cfg_overrides["mlflow_experiment_name"] = args.experiment

    cfg = Settings(**cfg_overrides)
    logger.info("Configuration: %s", cfg.model_dump())

    # Build data loaders
    logger.info("Loading AG News dataset...")
    train_loader, val_loader, test_loader, label_encoder = build_dataloaders(cfg)

    # Update num_labels from actual data
    cfg_dict = cfg.model_dump()
    cfg_dict["num_labels"] = len(label_encoder.classes_)
    cfg = Settings(**cfg_dict)

    # Build model
    logger.info("Initialising DistilBertClassifier (num_labels=%d)...", cfg.num_labels)
    model = DistilBertClassifier.from_config(cfg)
    if args.freeze_base:
        model.freeze_base_model()
        logger.info("Base model frozen — training classification head only")
    logger.info("Trainable parameters: %d", model.trainable_parameter_count())

    # Build and run trainer
    trainer = MLflowTrainer.from_config(
        cfg=cfg,
        model=model,
        train_dataset=train_loader.dataset,
        eval_dataset=val_loader.dataset,
        compute_metrics=compute_metrics,
        output_dir=args.output_dir,
    )
    trainer.log_confusion_matrix = not args.no_confusion_matrix

    logger.info("Starting training — experiment: %s", cfg.mlflow_experiment_name)
    train_result = trainer.train()

    # Final evaluation on test set
    logger.info("Running final evaluation on test set...")
    test_metrics = trainer.evaluate(eval_dataset=test_loader.dataset)
    logger.info("Test metrics: %s", test_metrics)

    logger.info("Training complete. Metrics: %s", train_result.metrics)


if __name__ == "__main__":
    main()
