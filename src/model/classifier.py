"""
DistilBERT-based multi-label text classifier.

Wraps HuggingFace's DistilBertModel with a custom classification head
suitable for multi-label classification tasks (e.g. AG News topic tagging).
The head replaces the default single linear layer with a two-layer MLP
plus dropout, which typically yields a small but consistent accuracy gain.

Usage::

    from src.config import Settings
    from src.model.classifier import DistilBertClassifier

    cfg = Settings()
    model = DistilBertClassifier.from_config(cfg)
    outputs = model(input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"])
    logits = outputs.logits          # (B, num_labels)
    pooled = outputs.pooled_output   # (B, hidden_size)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import DistilBertConfig, DistilBertModel, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from src.config import Settings


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class ClassifierOutput:
    """Container for classifier forward-pass outputs.

    Attributes:
        logits: Raw (pre-sigmoid) scores of shape ``(batch_size, num_labels)``.
        pooled_output: CLS-token representation after the projection layer,
            shape ``(batch_size, hidden_size)``.  Useful for visualisation /
            probing experiments.
        loss: Cross-entropy / BCE loss when ``labels`` are supplied.
    """

    logits: torch.Tensor
    pooled_output: torch.Tensor
    loss: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Classifier head
# ---------------------------------------------------------------------------

class ClassificationHead(nn.Module):
    """Two-layer MLP classification head with dropout and GELU activation.

    Architecture::

        Linear(hidden_size → hidden_size) → GELU → Dropout → Linear(hidden_size → num_labels)

    Args:
        hidden_size: Dimensionality of the encoder hidden states.
        num_labels: Number of output classes.
        dropout_prob: Dropout probability applied between the two linear layers.

    Example::

        head = ClassificationHead(hidden_size=768, num_labels=4, dropout_prob=0.2)
        logits = head(cls_token)   # (B, 4)
    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_labels)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier-uniform init for the projection layers."""
        nn.init.xavier_uniform_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            hidden_state: CLS token embedding, shape ``(B, hidden_size)``.

        Returns:
            Tuple of ``(logits, pooled_output)`` where logits are the raw
            pre-sigmoid scores and pooled_output is the hidden state after the
            first linear + activation (useful for probing).
        """
        x = self.dense(hidden_state)
        x = self.activation(x)
        pooled = x
        x = self.dropout(x)
        logits = self.out_proj(x)
        return logits, pooled


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class DistilBertClassifier(nn.Module):
    """Fine-tunable DistilBERT wrapper for multi-label text classification.

    The model extracts the CLS-token hidden state from DistilBERT and passes
    it through :class:`ClassificationHead` to produce per-class logits.  No
    sigmoid is applied here — callers should use ``BCEWithLogitsLoss`` during
    training and apply ``torch.sigmoid`` for inference.

    Args:
        pretrained_model_name: HuggingFace model hub ID or local path.
        num_labels: Number of target classes.
        dropout_prob: Dropout probability in the classification head.
        freeze_base: If ``True``, freeze all DistilBERT parameters and only
            train the classification head.  Useful for a warm-up phase.

    Example::

        model = DistilBertClassifier(
            pretrained_model_name="distilbert-base-uncased",
            num_labels=4,
            dropout_prob=0.2,
        )
        outputs = model(input_ids, attention_mask)
        # outputs.logits  →  (B, 4)
    """

    def __init__(
        self,
        pretrained_model_name: str = "distilbert-base-uncased",
        num_labels: int = 4,
        dropout_prob: float = 0.1,
        freeze_base: bool = False,
    ) -> None:
        super().__init__()

        self.num_labels = num_labels
        self.pretrained_model_name = pretrained_model_name

        self.distilbert: DistilBertModel = DistilBertModel.from_pretrained(
            pretrained_model_name
        )
        hidden_size: int = self.distilbert.config.hidden_size

        self.classifier = ClassificationHead(
            hidden_size=hidden_size,
            num_labels=num_labels,
            dropout_prob=dropout_prob,
        )

        if freeze_base:
            self.freeze_base_model()

    # ------------------------------------------------------------------
    # Class constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: Settings) -> "DistilBertClassifier":
        """Instantiate from a :class:`~src.config.Settings` object.

        Args:
            cfg: Application settings.  Uses ``cfg.model_name``,
                ``cfg.num_labels``, and ``cfg.dropout_prob``.

        Returns:
            Configured :class:`DistilBertClassifier` instance.

        Example::

            cfg = Settings()
            model = DistilBertClassifier.from_config(cfg)
        """
        return cls(
            pretrained_model_name=cfg.model_name,
            num_labels=cfg.num_labels,
            dropout_prob=cfg.dropout_prob,
        )

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def freeze_base_model(self) -> None:
        """Freeze all DistilBERT encoder parameters.

        Useful to train only the classification head for a warm-up phase
        before unfreezing the full model.
        """
        for param in self.distilbert.parameters():
            param.requires_grad = False

    def unfreeze_base_model(self) -> None:
        """Unfreeze all DistilBERT encoder parameters."""
        for param in self.distilbert.parameters():
            param.requires_grad = True

    def trainable_parameter_count(self) -> int:
        """Return the number of trainable parameters.

        Returns:
            Integer count of parameters with ``requires_grad=True``.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> ClassifierOutput:
        """Run a forward pass through DistilBERT and the classification head.

        Args:
            input_ids: Token IDs, shape ``(B, seq_len)``.
            attention_mask: Binary mask, shape ``(B, seq_len)``.  ``1`` for
                real tokens, ``0`` for padding.
            labels: Ground-truth binary label matrix, shape ``(B, num_labels)``,
                dtype ``float``.  When supplied, loss is computed via
                :class:`torch.nn.BCEWithLogitsLoss` and attached to the output.

        Returns:
            :class:`ClassifierOutput` with ``logits``, ``pooled_output``, and
            optionally ``loss``.

        Raises:
            ValueError: If ``labels`` shape does not match ``(B, num_labels)``.
        """
        encoder_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # DistilBERT returns (last_hidden_state,) — CLS token is at position 0
        cls_hidden: torch.Tensor = encoder_output.last_hidden_state[:, 0, :]  # (B, H)

        logits, pooled_output = self.classifier(cls_hidden)

        loss: Optional[torch.Tensor] = None
        if labels is not None:
            if labels.shape != (input_ids.size(0), self.num_labels):
                raise ValueError(
                    f"Expected labels shape ({input_ids.size(0)}, {self.num_labels}), "
                    f"got {tuple(labels.shape)}"
                )
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels.float())

        return ClassifierOutput(logits=logits, pooled_output=pooled_output, loss=loss)
