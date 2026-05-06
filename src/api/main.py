"""
FastAPI inference server for the LLM Text Classifier.

Endpoints
---------
POST /predict
    Classify a piece of text and return the top-k labels with scores.
GET  /health
    Liveness check -- confirms the ONNX engine is loaded.

Usage
-----
Start the server::

    uvicorn src.api.main:app --host 0.0.0.0 --port 8000

Or via Docker::

    docker compose up
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer

from src.api.schemas import HealthResponse, PredictRequest, PredictResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Runtime state -- populated during lifespan startup
# ---------------------------------------------------------------------------

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the ONNX engine and tokenizer once at server start-up.

    The engine is placed in the module-level _state dict so that request
    handlers can access it without re-loading per request.

    Yields
    ------
    None
        Control is yielded back to FastAPI; clean-up runs on shutdown.
    """
    import os

    model_dir = os.environ.get("MODEL_DIR", "artifacts/onnx")
    model_path = os.path.join(model_dir, "model.onnx")
    label_path = os.path.join(model_dir, "label_map.npy")
    tokenizer_path = os.environ.get("TOKENIZER_DIR", "artifacts/tokenizer")

    logger.info("Loading tokenizer from %s", tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    logger.info("Loading ONNX session from %s", model_path)
    providers = ort.get_available_providers()
    session = ort.InferenceSession(model_path, providers=providers)

    label_map: dict[int, str] = {}
    if os.path.exists(label_path):
        raw = np.load(label_path, allow_pickle=True).item()
        label_map = {int(k): str(v) for k, v in raw.items()}
    else:
        logger.warning("label_map.npy not found -- using numeric labels")

    _state["session"] = session
    _state["tokenizer"] = tokenizer
    _state["label_map"] = label_map
    _state["provider"] = providers[0]

    logger.info("Server ready. Provider: %s", providers[0])
    yield

    logger.info("Shutting down -- releasing ONNX session")
    _state.clear()


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LLM Text Classifier",
    description=(
        "Zero-shot and fine-tuned text classification via an ONNX-optimised "
        "transformer model. Accepts arbitrary text and returns top-k labels "
        "with confidence scores."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Return service liveness status and model-load state.

    Returns
    -------
    HealthResponse
        status is always "ok" when the endpoint is reachable.
        model_loaded is True only after the lifespan startup succeeded.

    Example
    -------
    .. code-block:: bash

        curl http://localhost:8000/health
    """
    model_loaded = "session" in _state
    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        device=_state.get("provider", "not loaded"),
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(request: PredictRequest) -> PredictResponse:
    """Classify input text and return the top-k labels with confidence scores.

    Parameters
    ----------
    request:
        text -- the raw string to classify (1-8192 characters).
        top_k -- number of labels to return (1-10, default 3).

    Returns
    -------
    PredictResponse
        labels     -- top-k class names ordered by descending score.
        scores     -- corresponding softmax probabilities.
        latency_ms -- server-side inference time in milliseconds.

    Raises
    ------
    HTTPException (503)
        If the ONNX session has not been loaded (startup failure).
    HTTPException (422)
        If the request payload fails Pydantic validation.

    Example
    -------
    .. code-block:: bash

        curl -X POST http://localhost:8000/predict \
             -H "Content-Type: application/json" \
             -d '{"text": "Federer wins Wimbledon again", "top_k": 3}'
    """
    if "session" not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    session: ort.InferenceSession = _state["session"]
    tokenizer = _state["tokenizer"]
    label_map: dict[int, str] = _state["label_map"]

    t0 = time.perf_counter()

    encoding = tokenizer(
        request.text,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=128,
    )

    input_feed = {
        "input_ids": encoding["input_ids"].astype(np.int64),
        "attention_mask": encoding["attention_mask"].astype(np.int64),
    }
    if "token_type_ids" in encoding:
        input_feed["token_type_ids"] = encoding["token_type_ids"].astype(np.int64)

    outputs = session.run(None, input_feed)
    logits: np.ndarray = outputs[0][0]  # shape: (num_classes,)

    e = np.exp(logits - logits.max())
    probs = e / e.sum()

    top_k = min(request.top_k, len(probs))
    top_indices = np.argsort(probs)[::-1][:top_k]

    labels = [label_map.get(int(i), str(i)) for i in top_indices]
    scores = [float(probs[i]) for i in top_indices]
    latency_ms = (time.perf_counter() - t0) * 1_000

    return PredictResponse(labels=labels, scores=scores, latency_ms=latency_ms)
