"""
CLI script: export a fine-tuned text classifier to ONNX and benchmark latency.

Usage
-----
    python scripts/export.py \
        --checkpoint artifacts/best_model.pt \
        --output     artifacts/model.onnx \
        --model-name distilbert-base-uncased \
        --num-labels 3 \
        [--max-seq-len 128] \
        [--batch-size 32] \
        [--benchmark-runs 20]

The script:
  1. Loads the PyTorch checkpoint.
  2. Exports to ONNX via :class:`src.inference.exporter.ModelExporter`.
  3. Benchmarks ONNX latency against PyTorch.
  4. Prints a side-by-side comparison table.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.engine import ONNXInferenceEngine
from src.inference.exporter import ModelExporter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    p = argparse.ArgumentParser(
        description="Export text-classifier to ONNX and benchmark inference."
    )
    p.add_argument("--checkpoint",     required=True, type=Path)
    p.add_argument("--output",         required=True, type=Path)
    p.add_argument("--model-name",     default="distilbert-base-uncased")
    p.add_argument("--num-labels",     type=int, default=2)
    p.add_argument("--max-seq-len",    type=int, default=128)
    p.add_argument("--batch-size",     type=int, default=32)
    p.add_argument("--benchmark-runs", type=int, default=20)
    p.add_argument("--atol",           type=float, default=1e-4)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def _benchmark_pytorch(
    model: torch.nn.Module,
    tokenizer,
    texts: List[str],
    max_seq_len: int,
    n_runs: int,
) -> dict:
    """Measure PyTorch CPU inference latency.

    Args:
        model: Eval-mode PyTorch model.
        tokenizer: Matching tokenizer.
        texts: Input texts for the timed batch.
        max_seq_len: Truncation length.
        n_runs: Number of timed repetitions.

    Returns:
        Dict with keys ``mean_ms``, ``p50_ms``, ``p95_ms``.
    """
    model.eval()
    device = next(model.parameters()).device
    enc = tokenizer(
        texts, padding="max_length", truncation=True,
        max_length=max_seq_len, return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in enc.items()}
    latencies: List[float] = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(**inputs)
            latencies.append((time.perf_counter() - t0) * 1000)
    arr = np.array(latencies)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms":  float(np.percentile(arr, 50)),
        "p95_ms":  float(np.percentile(arr, 95)),
    }


def _print_comparison(pt_stats: dict, onnx_stats: dict, batch_size: int) -> None:
    """Print a side-by-side latency comparison table.

    Args:
        pt_stats: Stats dict from :func:`_benchmark_pytorch`.
        onnx_stats: Stats dict from :meth:`ONNXInferenceEngine.benchmark`.
        batch_size: Samples per batch (for throughput computation).
    """
    header = f"{'Metric':<22}{'PyTorch':>14}{'ONNX':>14}{'Speedup':>12}"
    sep = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for key, label in [
        ("mean_ms", "Mean latency"),
        ("p50_ms",  "p50 latency"),
        ("p95_ms",  "p95 latency"),
    ]:
        pt_v, on_v = pt_stats[key], onnx_stats[key]
        speedup = pt_v / on_v if on_v else float("inf")
        print(f"  {label:<20}{pt_v:>12.2f}ms{on_v:>12.2f}ms{speedup:>10.2f}x")

    pt_tps   = batch_size * 1000 / pt_stats["mean_ms"]
    onnx_tps = batch_size * 1000 / onnx_stats["mean_ms"]
    print(f"  {'Throughput':<20}{pt_tps:>11.0f}/s{onnx_tps:>11.0f}/s{onnx_tps/pt_tps:>10.2f}x")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    """Entry point: load checkpoint, export to ONNX, benchmark, report."""
    args = parse_args(argv)

    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    logger.info("Loading tokenizer and model from %r...", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=args.num_labels
    )

    if args.checkpoint.exists():
        state = torch.load(args.checkpoint, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        elif isinstance(state, dict) and all(isinstance(k, str) for k in state):
            model.load_state_dict(state)
        logger.info("Loaded weights from %s", args.checkpoint)
    else:
        logger.warning("Checkpoint not found at %s - using random weights", args.checkpoint)

    exporter = ModelExporter(model, tokenizer, atol=args.atol)
    report = exporter.export(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        max_seq_len=args.max_seq_len,
    )
    print(report)

    logger.info(
        "Benchmarking: batch_size=%d, n_runs=%d...",
        args.batch_size, args.benchmark_runs,
    )
    sample_texts = [
        "This is a benchmark sentence for evaluating classifier latency."
    ] * args.batch_size

    id2label = {i: f"label_{i}" for i in range(args.num_labels)}
    engine = ONNXInferenceEngine(
        args.output, tokenizer, id2label, max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
    )

    onnx_stats = engine.benchmark(sample_texts, n_runs=args.benchmark_runs)
    pt_stats   = _benchmark_pytorch(
        model, tokenizer, sample_texts, args.max_seq_len, args.benchmark_runs
    )

    _print_comparison(pt_stats, onnx_stats, args.batch_size)


if __name__ == "__main__":
    main()
