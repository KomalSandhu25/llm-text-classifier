"""
Model package for llm-text-classifier.

Exports the core classifier, loss functions, and evaluation metrics used
throughout training and inference.
"""

from src.model.classifier import DistilBertClassifier, ClassifierOutput
from src.model.losses import MultiLabelLoss, WeightedMultiLabelLoss
from src.model.metrics import compute_metrics

__all__ = [
    "DistilBertClassifier",
    "ClassifierOutput",
    "MultiLabelLoss",
    "WeightedMultiLabelLoss",
    "compute_metrics",
]
