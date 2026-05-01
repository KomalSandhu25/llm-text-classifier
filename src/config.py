"""
Application configuration via environment variables.

All training hyperparameters and paths are centralised here so they are
reproducible, loggable to MLflow, and never hardcoded in training scripts.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "llm-text-classifier"

    # ── Model ─────────────────────────────────────────────────────────────────
    model_name: str = "distilbert-base-uncased"
    num_labels: int = 4

    # ── Tokenisation ──────────────────────────────────────────────────────────
    max_seq_length: int = Field(default=128, ge=16, le=512)

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size: int = Field(default=32, ge=1)
    num_epochs: int = Field(default=5, ge=1, le=100)
    learning_rate: float = Field(default=2e-5, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    warmup_ratio: float = Field(default=0.1, ge=0, le=1)
    fp16: bool = True
    gradient_accumulation_steps: int = 1
    seed: int = 42

    # ── Data ──────────────────────────────────────────────────────────────────
    dataset_name: str = "ag_news"
    data_dir: str = "./data"
    test_size: float = Field(default=0.1, gt=0, lt=1)
    val_size: float = Field(default=0.1, gt=0, lt=1)

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir: str = "./models"
    model_path: str = "./models/best_model.onnx"

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    @field_validator("learning_rate")
    @classmethod
    def lr_must_be_small(cls, v: float) -> float:
        if v > 0.1:
            raise ValueError("Learning rate looks too high for a fine-tuning job — did you mean something like 2e-5?")
        return v

    def as_mlflow_params(self) -> dict:
        """Return a flat dict suitable for mlflow.log_params()."""
        return {
            "model_name": self.model_name,
            "max_seq_length": self.max_seq_length,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "fp16": self.fp16,
            "seed": self.seed,
            "dataset": self.dataset_name,
        }

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def output_path(self) -> Path:
        p = Path(self.output_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


# Module-level singleton — import this everywhere
settings = Settings()
