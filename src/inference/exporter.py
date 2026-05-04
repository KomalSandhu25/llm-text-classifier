"""
ONNX export module for llm-text-classifier.

Converts a fine-tuned PyTorch text-classification model to ONNX format
with dynamic axes, validates numerical equivalence against the original
PyTorch model, and reports file-size reduction.

Typical usage
-------------
>>> from src.inference.exporter import ModelExporter
>>> exporter = ModelExporter(model, tokenizer, device="cpu")
>>> report = exporter.export("checkpoint.pt", "model.onnx")
>>> print(report)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


@dataclass
class ExportReport:
    """Summary statistics produced after a successful ONNX export.

    Attributes:
        onnx_path: Absolute path to the written ONNX file.
        pytorch_size_mb: On-disk size of the PyTorch checkpoint in MiB.
        onnx_size_mb: On-disk size of the exported ONNX file in MiB.
        size_reduction_pct: Percentage reduction in file size.
        max_abs_diff: Maximum absolute difference between PyTorch and ONNX
            logits on the validation batch.
        validation_passed: Whether ``max_abs_diff`` is within tolerance.
        export_time_s: Wall-clock time (seconds) for the export step.
    """

    onnx_path: str
    pytorch_size_mb: float
    onnx_size_mb: float
    size_reduction_pct: float
    max_abs_diff: float
    validation_passed: bool
    export_time_s: float
    extra_info: Dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:  # noqa: D105
        lines = [
            "=== ONNX Export Report ===",
            f"  ONNX path          : {self.onnx_path}",
            f"  PyTorch size       : {self.pytorch_size_mb:.2f} MiB",
            f"  ONNX size          : {self.onnx_size_mb:.2f} MiB",
            f"  Size reduction     : {self.size_reduction_pct:.1f}%",
            f"  Max |Delta logit|  : {self.max_abs_diff:.6f}",
            f"  Validation passed  : {self.validation_passed}",
            f"  Export time        : {self.export_time_s:.2f}s",
        ]
        for k, v in self.extra_info.items():
            lines.append(f"  {k:<20}: {v}")
        return "\n".join(lines)


class ModelExporter:
    """Export a fine-tuned HuggingFace-backed classifier to ONNX.

    The exporter creates a dummy batch, traces the forward pass with
    ``torch.onnx.export``, then validates that maximum absolute logit
    deviation between PyTorch and ONNX is within ``atol``.

    Args:
        model: Fine-tuned ``nn.Module`` in eval mode (or will be set to eval).
        tokenizer: Tokenizer compatible with the model.
        device: Device string or ``torch.device`` for tracing (default ``"cpu"``).
        atol: Absolute tolerance for logit comparison (default ``1e-4``).
        opset_version: ONNX opset to target (default ``14``).

    Example:
        >>> exporter = ModelExporter(model, tokenizer)
        >>> report = exporter.export("weights.pt", "model.onnx")
        >>> assert report.validation_passed
    """

    _DUMMY_TEXT = "The quick brown fox jumps over the lazy dog."

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        device: Union[str, torch.device] = "cpu",
        atol: float = 1e-4,
        opset_version: int = 14,
    ) -> None:
        self.model = model.eval().to(device)
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.atol = atol
        self.opset_version = opset_version

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export(
        self,
        checkpoint_path: Union[str, Path],
        output_path: Union[str, Path],
        validation_texts: Optional[List[str]] = None,
        max_seq_len: int = 128,
    ) -> ExportReport:
        """Export the model to ONNX and return a validation report.

        Args:
            checkpoint_path: Path to the ``.pt`` checkpoint file (used only
                for size comparison; weights are already loaded in ``self.model``).
            output_path: Destination path for the ``.onnx`` file.
            validation_texts: Texts used for numerical equivalence checking.
                Defaults to a built-in dummy sentence.
            max_seq_len: Maximum sequence length for the dummy/validation batch.

        Returns:
            ExportReport with size and validation details.

        Raises:
            RuntimeError: If validation fails (logit deviation exceeds ``atol``).
        """
        checkpoint_path = Path(checkpoint_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if validation_texts is None:
            validation_texts = [self._DUMMY_TEXT]

        dummy_inputs = self._tokenize(validation_texts, max_seq_len)

        logger.info("Exporting model to ONNX (opset %d)...", self.opset_version)
        t0 = time.perf_counter()
        self._run_export(dummy_inputs, output_path)
        export_time = time.perf_counter() - t0
        logger.info("Export complete in %.2fs", export_time)

        max_diff = self._validate(dummy_inputs, output_path)
        passed = max_diff <= self.atol

        pytorch_mb = self._file_mb(checkpoint_path) if checkpoint_path.exists() else 0.0
        onnx_mb = self._file_mb(output_path)
        reduction = (1 - onnx_mb / pytorch_mb) * 100 if pytorch_mb else 0.0

        report = ExportReport(
            onnx_path=str(output_path.resolve()),
            pytorch_size_mb=pytorch_mb,
            onnx_size_mb=onnx_mb,
            size_reduction_pct=reduction,
            max_abs_diff=max_diff,
            validation_passed=passed,
            export_time_s=export_time,
        )
        logger.info("\n%s", report)

        if not passed:
            raise RuntimeError(
                f"ONNX validation failed: max |Delta logit| = {max_diff:.6f} > atol={self.atol}"
            )
        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _tokenize(
        self, texts: List[str], max_length: int
    ) -> Dict[str, torch.Tensor]:
        """Tokenise *texts* and move tensors to self.device."""
        enc = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    def _run_export(
        self, dummy_inputs: Dict[str, torch.Tensor], output_path: Path
    ) -> None:
        """Invoke torch.onnx.export with dynamic batch + sequence axes."""
        input_names = list(dummy_inputs.keys())
        dynamic_axes: Dict[str, Dict[int, str]] = {
            name: {0: "batch_size", 1: "sequence_length"} for name in input_names
        }
        dynamic_axes["logits"] = {0: "batch_size"}

        with torch.no_grad():
            torch.onnx.export(
                self.model,
                args=tuple(dummy_inputs.values()),
                f=str(output_path),
                input_names=input_names,
                output_names=["logits"],
                dynamic_axes=dynamic_axes,
                opset_version=self.opset_version,
                do_constant_folding=True,
                export_params=True,
            )

    @torch.no_grad()
    def _validate(
        self, inputs: Dict[str, torch.Tensor], onnx_path: Path
    ) -> float:
        """Return max absolute difference between PyTorch and ONNX logits."""
        import onnxruntime as ort

        pt_logits = self.model(**inputs).logits.cpu().numpy()

        sess = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        ort_inputs = {k: v.cpu().numpy() for k, v in inputs.items()}
        (onnx_logits,) = sess.run(None, ort_inputs)

        diff = float(np.max(np.abs(pt_logits - onnx_logits)))
        logger.debug("Max |Delta logit| between PyTorch and ONNX: %.8f", diff)
        return diff

    @staticmethod
    def _file_mb(path: Path) -> float:
        """Return file size in MiB, or 0 if file does not exist."""
        try:
            return os.path.getsize(path) / (1024 ** 2)
        except FileNotFoundError:
            return 0.0
