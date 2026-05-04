"""
Unit tests for the inference package.

Uses lightweight mocks to avoid loading real models or ONNX runtimes,
keeping CI fast while exercising all critical code paths.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def id2label() -> Dict[int, str]:
    return {0: "negative", 1: "neutral", 2: "positive"}


@pytest.fixture()
def mock_tokenizer():
    """Minimal tokenizer mock that returns fixed np arrays."""
    tok = MagicMock()
    tok.side_effect = lambda texts, **kwargs: {
        "input_ids":      np.ones((len(texts), 8), dtype=np.int64),
        "attention_mask": np.ones((len(texts), 8), dtype=np.int64),
    }
    return tok


def _make_ort_session(logits: np.ndarray):
    """Return a mock onnxruntime.InferenceSession that yields *logits*."""
    session = MagicMock()
    session.get_inputs.return_value = [
        types.SimpleNamespace(name="input_ids"),
        types.SimpleNamespace(name="attention_mask"),
    ]
    session.run.return_value = [logits]
    return session


# ---------------------------------------------------------------------------
# ONNXInferenceEngine tests
# ---------------------------------------------------------------------------

class TestONNXInferenceEngine:
    """Tests for ONNXInferenceEngine."""

    def _build_engine(self, logits, mock_tokenizer, id2label, batch_size=32):
        from src.inference.engine import ONNXInferenceEngine

        with patch("onnxruntime.InferenceSession", return_value=_make_ort_session(logits)):
            with patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"]):
                engine = ONNXInferenceEngine(
                    "dummy.onnx", mock_tokenizer, id2label, batch_size=batch_size
                )
        return engine

    def test_predict_single_text_returns_one_result(self, mock_tokenizer, id2label):
        """Single input should yield exactly one PredictionResult."""
        logits = np.array([[0.1, 0.2, 3.0]])
        engine = self._build_engine(logits, mock_tokenizer, id2label)

        results = engine.predict(["I love this!"])

        assert len(results) == 1
        assert results[0].label == "positive"
        assert results[0].label_id == 2
        assert 0.0 < results[0].confidence <= 1.0

    def test_predict_batch_correct_labels(self, mock_tokenizer, id2label):
        """All samples in a batch should get the highest-logit class."""
        n = 5
        logits = np.zeros((n, 3))
        logits[:, 1] = 10.0
        engine = self._build_engine(logits, mock_tokenizer, id2label)

        results = engine.predict(["text"] * n)

        assert len(results) == n
        assert all(r.label == "neutral" for r in results)

    def test_predict_splits_into_batches(self, mock_tokenizer, id2label):
        """Inputs exceeding batch_size should be split across multiple ORT calls."""
        batch_size = 4
        n_texts = 10
        from src.inference.engine import ONNXInferenceEngine

        session = _make_ort_session(np.zeros((batch_size, 3)))

        def flexible_run(_names, inputs):
            b = inputs["input_ids"].shape[0]
            out = np.zeros((b, 3))
            out[:, 0] = 5.0
            return [out]

        session.run.side_effect = flexible_run

        with patch("onnxruntime.InferenceSession", return_value=session):
            with patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"]):
                engine = ONNXInferenceEngine(
                    "dummy.onnx", mock_tokenizer, id2label, batch_size=batch_size
                )

        results = engine.predict(["t"] * n_texts)
        assert len(results) == n_texts

    def test_predict_empty_raises_value_error(self, mock_tokenizer, id2label):
        """Passing an empty list should raise ValueError."""
        logits = np.zeros((1, 3))
        engine = self._build_engine(logits, mock_tokenizer, id2label)

        with pytest.raises(ValueError, match="non-empty"):
            engine.predict([])

    def test_all_scores_sum_to_one(self, mock_tokenizer, id2label):
        """Softmax probabilities across all classes should sum to 1."""
        logits = np.array([[1.0, 2.0, 0.5]])
        engine = self._build_engine(logits, mock_tokenizer, id2label)
        result = engine.predict(["test"])[0]

        assert abs(sum(result.all_scores.values()) - 1.0) < 1e-5

    def test_confidence_matches_all_scores(self, mock_tokenizer, id2label):
        """result.confidence should equal all_scores[result.label]."""
        logits = np.array([[0.0, 5.0, 1.0]])
        engine = self._build_engine(logits, mock_tokenizer, id2label)
        result = engine.predict(["test"])[0]

        assert abs(result.confidence - result.all_scores[result.label]) < 1e-6

    def test_softmax_numerically_stable_large_logits(self):
        """Softmax of very large logits should not produce NaN or Inf."""
        from src.inference.engine import ONNXInferenceEngine

        large = np.array([[1000.0, 1001.0, 999.0]])
        out = ONNXInferenceEngine._softmax(large)

        assert np.allclose(out.sum(axis=-1), 1.0)
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))

    def test_benchmark_returns_required_keys(self, mock_tokenizer, id2label):
        """benchmark() should return the four expected metric keys."""
        logits = np.array([[0.1, 0.9]])
        id2label_2 = {0: "neg", 1: "pos"}
        engine = self._build_engine(logits, mock_tokenizer, id2label_2)

        def flexible_run(_names, inputs):
            b = inputs["input_ids"].shape[0]
            return [np.tile([0.1, 0.9], (b, 1))]

        engine._session.run.side_effect = flexible_run

        stats = engine.benchmark(["hello"] * 4, n_runs=3)
        assert set(stats.keys()) == {"mean_ms", "p50_ms", "p95_ms", "throughput_samples_per_s"}
        assert stats["mean_ms"] > 0

    def test_latency_ms_positive(self, mock_tokenizer, id2label):
        """Reported latency should always be a positive float."""
        logits = np.array([[1.0, 0.0, 0.0]])
        engine = self._build_engine(logits, mock_tokenizer, id2label)
        result = engine.predict(["latency test"])[0]
        assert result.latency_ms > 0


# ---------------------------------------------------------------------------
# ModelExporter tests
# ---------------------------------------------------------------------------

class TestModelExporter:
    """Tests for ModelExporter."""

    def _make_model(self, logits_np: np.ndarray):
        """Build a trivial nn.Module that returns a namespace with .logits."""
        import torch
        import torch.nn as nn

        class FakeOutput:
            def __init__(self, logits):
                self.logits = logits

        logits_t = torch.tensor(logits_np, dtype=torch.float32)

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1, 1, bias=False)

            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                return FakeOutput(logits_t.expand(input_ids.shape[0], -1))

        return FakeModel()

    def test_export_report_validation_passed(self, mock_tokenizer, tmp_path):
        """ExportReport should mark validation_passed=True when logits match."""
        import torch
        from src.inference.exporter import ModelExporter

        logits_np = np.array([[0.3, 0.7]])
        model = self._make_model(logits_np)

        dummy_onnx = tmp_path / "model.onnx"
        dummy_onnx.write_bytes(b"fake")

        ort_session = MagicMock()
        ort_session.run.return_value = [logits_np]

        with patch("torch.onnx.export", return_value=None):
            with patch("onnxruntime.InferenceSession", return_value=ort_session):
                exporter = ModelExporter(model, mock_tokenizer, device="cpu", atol=1.0)
                report = exporter.export(
                    checkpoint_path=tmp_path / "weights.pt",
                    output_path=dummy_onnx,
                    max_seq_len=8,
                )

        assert report.validation_passed
        assert report.onnx_path == str(dummy_onnx.resolve())
        assert report.max_abs_diff < 1.0

    def test_export_report_str_contains_key_fields(self):
        """ExportReport.__str__ should contain ONNX path and validation result."""
        from src.inference.exporter import ExportReport

        r = ExportReport(
            onnx_path="/tmp/model.onnx",
            pytorch_size_mb=50.0,
            onnx_size_mb=25.0,
            size_reduction_pct=50.0,
            max_abs_diff=1e-6,
            validation_passed=True,
            export_time_s=3.14,
        )
        s = str(r)
        assert "ONNX Export Report" in s
        assert "/tmp/model.onnx" in s
        assert "True" in s

    def test_export_raises_when_validation_fails(self, mock_tokenizer, tmp_path):
        """ModelExporter.export should raise RuntimeError when logits diverge."""
        import torch
        from src.inference.exporter import ModelExporter

        logits_np = np.array([[0.3, 0.7]])
        model = self._make_model(logits_np)

        dummy_onnx = tmp_path / "model.onnx"
        dummy_onnx.write_bytes(b"fake")

        # Return logits that differ by 1.0 (beyond atol=1e-4)
        bad_logits = logits_np + 1.0
        ort_session = MagicMock()
        ort_session.run.return_value = [bad_logits]

        with patch("torch.onnx.export", return_value=None):
            with patch("onnxruntime.InferenceSession", return_value=ort_session):
                exporter = ModelExporter(model, mock_tokenizer, device="cpu", atol=1e-4)
                with pytest.raises(RuntimeError, match="ONNX validation failed"):
                    exporter.export(
                        checkpoint_path=tmp_path / "weights.pt",
                        output_path=dummy_onnx,
                        max_seq_len=8,
                    )
