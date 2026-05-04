"""
Optimised ONNX inference engine for llm-text-classifier.

Wraps an exported ONNX model with HuggingFace tokenisation to provide
a clean, high-throughput batch-inference interface. Measures per-batch
latency and exposes confidence scores alongside predicted labels.

Typical usage
-------------
>>> engine = ONNXInferenceEngine("model.onnx", tokenizer, id2label)
>>> results = engine.predict(["Great product!", "Terrible experience."])
>>> for r in results:
...     print(r.label, f"{r.confidence:.2%}")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictionResult:
    """Single-sample classification result.

    Attributes:
        text: The original input text.
        label: Predicted class label string.
        label_id: Integer class index.
        confidence: Softmax probability of the predicted class (0-1).
        all_scores: Full probability distribution over all classes.
        latency_ms: Inference latency for the batch containing this sample (ms).
    """

    text: str
    label: str
    label_id: int
    confidence: float
    all_scores: Dict[str, float]
    latency_ms: float


class ONNXInferenceEngine:
    """Low-latency text classifier backed by an ONNX runtime session.

    Tokenises input texts, runs a single ONNX session call per batch,
    and returns structured :class:`PredictionResult` objects.

    Args:
        onnx_path: Path to the exported ``.onnx`` model file.
        tokenizer: HuggingFace tokenizer matching the model.
        id2label: Mapping from integer class index to label string, e.g.
            ``{0: "negative", 1: "neutral", 2: "positive"}``.
        max_seq_len: Maximum tokenisation length (default ``128``).
        batch_size: Maximum samples per ONNX forward pass (default ``32``).
        providers: ONNX Runtime execution providers in priority order.
            Defaults to ``["CUDAExecutionProvider", "CPUExecutionProvider"]``.

    Example:
        >>> engine = ONNXInferenceEngine(
        ...     "artifacts/model.onnx",
        ...     tokenizer,
        ...     id2label={0: "neg", 1: "pos"},
        ... )
        >>> results = engine.predict(["I love this!", "Waste of money."])
        >>> results[0].label
        "pos"
    """

    def __init__(
        self,
        onnx_path: Union[str, Path],
        tokenizer: PreTrainedTokenizerBase,
        id2label: Dict[int, str],
        max_seq_len: int = 128,
        batch_size: int = 32,
        providers: Optional[List[str]] = None,
    ) -> None:
        import onnxruntime as ort

        self.tokenizer = tokenizer
        self.id2label = id2label
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

        if providers is None:
            available = ort.get_available_providers()
            providers = [
                p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if p in available
            ]

        self._session = ort.InferenceSession(str(onnx_path), providers=providers)
        self._input_names = [inp.name for inp in self._session.get_inputs()]

        logger.info(
            "ONNXInferenceEngine loaded: %s  |  providers=%s  |  inputs=%s",
            onnx_path,
            providers,
            self._input_names,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, texts: Sequence[str]) -> List[PredictionResult]:
        """Classify a sequence of texts.

        Internally splits *texts* into batches of :attr:`batch_size` and
        concatenates results in order.

        Args:
            texts: One or more raw text strings to classify.

        Returns:
            List of :class:`PredictionResult` objects, one per input text,
            in the same order as *texts*.

        Raises:
            ValueError: If *texts* is empty.
        """
        if not texts:
            raise ValueError("texts must be non-empty")

        all_results: List[PredictionResult] = []
        for batch_start in range(0, len(texts), self.batch_size):
            batch = list(texts[batch_start : batch_start + self.batch_size])
            all_results.extend(self._predict_batch(batch))
        return all_results

    def benchmark(
        self,
        texts: Sequence[str],
        n_runs: int = 10,
    ) -> Dict[str, float]:
        """Measure mean and p95 inference latency.

        Args:
            texts: Sample texts used for the benchmark.
            n_runs: Number of timed forward passes (default ``10``).

        Returns:
            Dictionary with keys ``mean_ms``, ``p50_ms``, ``p95_ms``,
            ``throughput_samples_per_s``.

        Example:
            >>> stats = engine.benchmark(["hello world"] * 64)
            >>> print(f"p95 latency: {stats['p95_ms']:.1f}ms")
        """
        latencies: List[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.predict(list(texts))
            latencies.append((time.perf_counter() - t0) * 1000)

        arr = np.array(latencies)
        mean_ms = float(arr.mean())
        return {
            "mean_ms": mean_ms,
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "throughput_samples_per_s": len(texts) * 1000 / mean_ms,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _predict_batch(self, texts: List[str]) -> List[PredictionResult]:
        """Run a single ONNX forward pass for *texts* and parse results."""
        enc = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="np",
        )

        ort_inputs = {name: enc[name] for name in self._input_names if name in enc}

        t0 = time.perf_counter()
        (logits,) = self._session.run(None, ort_inputs)
        latency_ms = (time.perf_counter() - t0) * 1000

        probs = self._softmax(logits)
        results: List[PredictionResult] = []
        for i, text in enumerate(texts):
            label_id = int(np.argmax(probs[i]))
            label = self.id2label.get(label_id, str(label_id))
            all_scores = {
                self.id2label.get(j, str(j)): float(p)
                for j, p in enumerate(probs[i])
            }
            results.append(
                PredictionResult(
                    text=text,
                    label=label,
                    label_id=label_id,
                    confidence=float(probs[i, label_id]),
                    all_scores=all_scores,
                    latency_ms=latency_ms,
                )
            )
        return results

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Numerically stable row-wise softmax."""
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=-1, keepdims=True)
