"""Dataset loading and DataLoader construction for AG News.

This module is the single entry point for the data pipeline.  It downloads the
AG News dataset from the HuggingFace Hub, applies text preprocessing, converts
labels to multi-hot vectors, and returns ``torch.utils.data.DataLoader``
instances ready for training.

AG News has four classes:
    0 – World
    1 – Sports
    2 – Business
    3 – Sci/Tech

Although AG News is inherently single-label, the pipeline is built around
multi-label tensors (shape ``[num_labels]``) so the same ``DataLoader`` can
back a ``BCEWithLogitsLoss`` training loop without modification.

Example::

    from src.data.loaders import load_ag_news_dataloaders

    loaders = load_ag_news_dataloaders(batch_size=32, val_split=0.1)
    for batch in loaders["train"]:
        input_ids = batch["input_ids"]      # (B, L)
        labels    = batch["labels"]          # (B, 4)  — multi-hot floats
        break
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.data.preprocessor import TextPreprocessor, make_preprocessor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AG_NEWS_LABELS: list[str] = ["World", "Sports", "Business", "Sci/Tech"]
NUM_LABELS: int = len(AG_NEWS_LABELS)

# HuggingFace dataset identifier
_HF_DATASET: str = "ag_news"

# Default tokeniser (can be overridden via config)
_DEFAULT_MODEL: str = "distilbert-base-uncased"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Parameters controlling dataset loading and preprocessing.

    Attributes:
        model_name_or_path: HuggingFace model identifier used to instantiate
            the tokeniser.
        max_seq_length: Maximum token sequence length (padding / truncation).
        batch_size: Number of samples per DataLoader batch.
        val_split: Fraction of the *training* split reserved for validation.
        num_workers: DataLoader worker processes.
        seed: Random seed used for train/val split.
        max_train_samples: Cap the training set (useful for quick experiments).
            ``None`` means use the full set.
        max_eval_samples: Cap validation and test sets.  ``None`` means use
            the full set.
        preprocessor_max_chars: Character truncation applied before tokenisation.
    """

    model_name_or_path: str = _DEFAULT_MODEL
    max_seq_length: int = 128
    batch_size: int = 32
    val_split: float = 0.1
    num_workers: int = 0          # 0 = main process; safe default for all OS
    seed: int = 42
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    preprocessor_max_chars: Optional[int] = None


# ---------------------------------------------------------------------------
# Torch Dataset wrapper
# ---------------------------------------------------------------------------

class AGNewsDataset(Dataset):
    """Wrapper around a HuggingFace ``Dataset`` that tokenises on the fly.

    Each item is a dict with keys:
        - ``input_ids``      – ``torch.LongTensor`` of shape ``(max_seq_length,)``
        - ``attention_mask`` – ``torch.LongTensor`` of shape ``(max_seq_length,)``
        - ``labels``         – ``torch.FloatTensor`` of shape ``(num_labels,)``

    Args:
        hf_dataset: A HuggingFace ``Dataset`` with at least ``text`` and
            ``label`` columns.
        tokeniser: Pre-loaded tokeniser for the target model.
        preprocessor: Text cleaner applied *before* tokenisation.
        max_seq_length: Maximum token sequence length.
        num_labels: Number of output classes.

    Example::

        from datasets import load_dataset
        from transformers import AutoTokenizer

        raw = load_dataset("ag_news", split="train")
        tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        ds  = AGNewsDataset(raw, tok, TextPreprocessor(), max_seq_length=128)
        sample = ds[0]
        # {"input_ids": tensor([...]), "attention_mask": tensor([...]), "labels": tensor([...])}
    """

    def __init__(
        self,
        hf_dataset,
        tokeniser: PreTrainedTokenizerBase,
        preprocessor: TextPreprocessor,
        max_seq_length: int = 128,
        num_labels: int = NUM_LABELS,
    ) -> None:
        self._data = hf_dataset
        self._tokeniser = tokeniser
        self._preprocessor = preprocessor
        self._max_seq_length = max_seq_length
        self._num_labels = num_labels

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self._data[idx]
        text: str = self._preprocessor.clean(row["text"])
        label: int = row["label"]  # 0-indexed integer

        encoding = self._tokeniser(
            text,
            max_length=self._max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Convert single-label integer to multi-hot float vector
        label_vec = torch.zeros(self._num_labels, dtype=torch.float32)
        label_vec[label] = 1.0

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": label_vec,
        }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def load_ag_news_dataloaders(
    config: Optional[DataConfig] = None,
    **kwargs,
) -> Dict[str, DataLoader]:
    """Download AG News, preprocess, and return train/val/test DataLoaders.

    The AG News test split (7 600 samples) is used as-is.  The training split
    (120 000 samples) is further divided into train and validation sets using
    ``val_split``.

    Args:
        config: ``DataConfig`` instance.  When ``None``, keyword arguments are
            forwarded to ``DataConfig``.
        **kwargs: Passed to ``DataConfig`` when *config* is ``None``.

    Returns:
        Dictionary with keys ``"train"``, ``"val"``, and ``"test"``, each
        mapping to a ``torch.utils.data.DataLoader``.

    Raises:
        ValueError: If ``val_split`` is not in ``(0, 1)``.

    Example::

        loaders = load_ag_news_dataloaders(batch_size=64, max_seq_length=256)
        print(len(loaders["train"].dataset))  # ≈ 108 000
    """
    if config is None:
        config = DataConfig(**kwargs)

    if not 0.0 < config.val_split < 1.0:
        raise ValueError(
            f"val_split must be in (0, 1), got {config.val_split}"
        )

    logger.info("Loading AG News dataset from HuggingFace Hub …")
    raw: DatasetDict = load_dataset(_HF_DATASET)  # type: ignore[assignment]

    # ---- train / val split -----------------------------------------------
    train_val = raw["train"].train_test_split(
        test_size=config.val_split,
        seed=config.seed,
    )
    train_hf = train_val["train"]
    val_hf   = train_val["test"]
    test_hf  = raw["test"]

    # ---- optional sample caps --------------------------------------------
    if config.max_train_samples is not None:
        train_hf = train_hf.select(range(min(config.max_train_samples, len(train_hf))))
    if config.max_eval_samples is not None:
        val_hf  = val_hf.select(range(min(config.max_eval_samples, len(val_hf))))
        test_hf = test_hf.select(range(min(config.max_eval_samples, len(test_hf))))

    logger.info(
        "Split sizes — train: %d  val: %d  test: %d",
        len(train_hf), len(val_hf), len(test_hf),
    )

    # ---- shared components -----------------------------------------------
    preprocessor = make_preprocessor(
        max_chars=config.preprocessor_max_chars,
    )

    logger.info("Loading tokeniser: %s", config.model_name_or_path)
    tokeniser = AutoTokenizer.from_pretrained(config.model_name_or_path)

    # ---- Torch datasets --------------------------------------------------
    def _make_dataset(hf_split) -> AGNewsDataset:
        return AGNewsDataset(
            hf_dataset=hf_split,
            tokeniser=tokeniser,
            preprocessor=preprocessor,
            max_seq_length=config.max_seq_length,
        )

    train_ds = _make_dataset(train_hf)
    val_ds   = _make_dataset(val_hf)
    test_ds  = _make_dataset(test_hf)

    # ---- DataLoaders -----------------------------------------------------
    common_loader_kwargs = dict(
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    loaders: Dict[str, DataLoader] = {
        "train": DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            **common_loader_kwargs,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            **common_loader_kwargs,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=config.batch_size,
            shuffle=False,
            **common_loader_kwargs,
        ),
    }

    logger.info("DataLoaders ready.")
    return loaders


# ---------------------------------------------------------------------------
# Label utilities
# ---------------------------------------------------------------------------

def label_index_to_name(idx: int) -> str:
    """Return the human-readable label name for an AG News class index.

    Args:
        idx: Integer class index in ``[0, 3]``.

    Returns:
        Label name string (e.g. ``"Sports"``).

    Raises:
        IndexError: If *idx* is out of range.
    """
    return AG_NEWS_LABELS[idx]


def multihot_to_names(multihot: torch.Tensor, threshold: float = 0.5) -> list[str]:
    """Convert a multi-hot probability tensor to a list of label names.

    Args:
        multihot: Float tensor of shape ``(num_labels,)`` with values in
            ``[0, 1]``.
        threshold: Minimum value for a label to be considered active.

    Returns:
        Sorted list of active label names.

    Example::

        t = torch.tensor([0.9, 0.1, 0.05, 0.8])
        multihot_to_names(t)
        # ["Sci/Tech", "World"]
    """
    return [
        AG_NEWS_LABELS[i]
        for i, score in enumerate(multihot.tolist())
        if score >= threshold
    ]
