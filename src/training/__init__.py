"""
Training package for llm-text-classifier.

Exports the MLflow-integrated Trainer subclass and training callbacks.
"""

from src.training.trainer import MLflowTrainer
from src.training.callbacks import EarlyStoppingCallback, MLflowLoggingCallback

__all__ = [
    "MLflowTrainer",
    "EarlyStoppingCallback",
    "MLflowLoggingCallback",
]
