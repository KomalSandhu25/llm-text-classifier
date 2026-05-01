"""
TextClassificationDataset — wraps HuggingFace datasets for DistilBERT fine-tuning.

Handles:
  - Loading AG News (or any HuggingFace text classification dataset)
  - Tokenisation with padding and truncation
  - Label encoding / decoding
  - Returning PyTorch tensors compatible with HuggingFace Trainer
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from datasets import load_dataset, DatasetDict
from loguru import logger
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from src.config import settings


# ── AG News label mapping (4-class) ──────────────────────────────────────────
AG_NEWS_LABELS: Dict[int, str] = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Science/Technology",
}


def label_encoder(label_id: int, label_map: Dict[int, str] = AG_NEWS_LABELS) -> str:
    """Convert an integer label to its human-readable class name."""
    return label_map.get(label_id, f"Unknown({label_id})")


class TextClassificationDataset(Dataset):
    """
    PyTorch Dataset for single-label text classification.

    Each sample returns:
        input_ids      : (seq_len,) LongTensor
        attention_mask : (seq_len,) LongTensor
        labels         : () LongTensor (scalar class index)

    Example:
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        dataset   = TextClassificationDataset.from_hf_split("train", tokenizer)
        sample    = dataset[0]
        print(sample["input_ids"].shape)   # torch.Size([128])
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = settings.max_seq_length,
    ) -> None:
        if len(texts) != len(labels):
            raise ValueError(
                f"texts and labels must have the same length, "
                f"got {len(texts)} vs {len(labels)}"
            )
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    # ── Dataset protocol ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def from_hf_split(
        cls,
        split: str,
        tokenizer: PreTrainedTokenizerBase,
        dataset_name: str = settings.dataset_name,
        max_length: int = settings.max_seq_length,
        max_samples: Optional[int] = None,
    ) -> "TextClassificationDataset":
        """
        Load a named HuggingFace dataset split and return a ready-to-use Dataset.

        Args:
            split:        "train", "test", or a slice like "train[:1000]"
            tokenizer:    Pre-loaded HuggingFace tokenizer
            dataset_name: HuggingFace dataset identifier (default: ag_news)
            max_length:   Maximum token sequence length
            max_samples:  Cap the number of samples (useful for quick tests)
        """
        logger.info(f"Loading {dataset_name}/{split}…")
        raw = load_dataset(dataset_name, split=split)

        if max_samples:
            raw = raw.select(range(min(max_samples, len(raw))))

        texts  = raw["text"]
        labels = raw["label"]

        logger.info(
            f"Loaded {len(texts)} samples from {dataset_name}/{split} "
            f"| classes: {sorted(set(labels))}"
        )
        return cls(texts, labels, tokenizer, max_length)

    @classmethod
    def from_hf_dataset_dict(
        cls,
        tokenizer: PreTrainedTokenizerBase,
        dataset_name: str = settings.dataset_name,
    ) -> Dict[str, "TextClassificationDataset"]:
        """
        Load train + test splits at once.

        Returns a dict {"train": ..., "test": ...} for convenient unpacking.
        """
        raw: DatasetDict = load_dataset(dataset_name)
        result = {}
        for split_name, split_data in raw.items():
            result[split_name] = cls(
                texts=split_data["text"],
                labels=split_data["label"],
                tokenizer=tokenizer,
            )
        return result
