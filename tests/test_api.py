"""
Integration tests for the FastAPI inference server.

These tests use httpx.AsyncClient with the ASGI transport so they run
against the real application without needing a live server. The ONNX session
is patched with a lightweight stub so the suite stays fast and CI-friendly.

Run::

    pytest tests/test_api.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import onnxruntime as ort
from unittest.mock import MagicMock

from httpx import ASGITransport, AsyncClient

from src.api.main import app, _state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_session(num_classes: int = 4) -> MagicMock:
    """Return a mock ort.InferenceSession that returns random logits."""
    mock = MagicMock(spec=ort.InferenceSession)
    logits = np.random.randn(1, num_classes).astype(np.float32)
    mock.run.return_value = [logits]
    return mock


LABEL_MAP = {0: "sports", 1: "politics", 2: "technology", 3: "entertainment"}


@pytest.fixture(autouse=True)
def patch_state():
    """Pre-populate _state so the lifespan is bypassed in tests."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    session = _make_mock_session(num_classes=4)

    _state["session"] = session
    _state["tokenizer"] = tokenizer
    _state["label_map"] = LABEL_MAP
    _state["provider"] = "CPUExecutionProvider"

    yield

    _state.clear()


@pytest.fixture
async def client():
    """Provide an async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_health_ok(client: AsyncClient) -> None:
    """GET /health returns 200 with model_loaded=True when state is populated."""
    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["device"] == "CPUExecutionProvider"


@pytest.mark.anyio
async def test_health_without_model() -> None:
    """GET /health returns model_loaded=False when the ONNX session is absent."""
    _state.clear()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is False


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_predict_default_top_k(client: AsyncClient) -> None:
    """POST /predict returns 3 labels by default."""
    response = await client.post("/predict", json={"text": "The election results are in."})
    assert response.status_code == 200
    body = response.json()
    assert len(body["labels"]) == 3
    assert len(body["scores"]) == 3
    assert all(isinstance(s, float) for s in body["scores"])
    assert body["latency_ms"] >= 0.0


@pytest.mark.anyio
async def test_predict_custom_top_k(client: AsyncClient) -> None:
    """POST /predict respects the top_k parameter."""
    response = await client.post(
        "/predict",
        json={"text": "The new iPhone features a faster chip.", "top_k": 2},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["labels"]) == 2
    assert len(body["scores"]) == 2


@pytest.mark.anyio
async def test_predict_labels_are_strings(client: AsyncClient) -> None:
    """Labels are resolved through the label map and returned as strings."""
    response = await client.post(
        "/predict", json={"text": "Goal! The home team scores.", "top_k": 4}
    )
    assert response.status_code == 200
    body = response.json()
    for label in body["labels"]:
        assert isinstance(label, str)
        assert label in LABEL_MAP.values()


@pytest.mark.anyio
async def test_predict_empty_text_rejected(client: AsyncClient) -> None:
    """POST /predict with empty text returns 422 Unprocessable Entity."""
    response = await client.post("/predict", json={"text": "", "top_k": 1})
    assert response.status_code == 422


@pytest.mark.anyio
async def test_predict_top_k_out_of_range(client: AsyncClient) -> None:
    """POST /predict with top_k=0 returns 422."""
    response = await client.post("/predict", json={"text": "Hello world", "top_k": 0})
    assert response.status_code == 422


@pytest.mark.anyio
async def test_predict_503_when_model_missing() -> None:
    """POST /predict returns 503 when the ONNX session is not loaded."""
    saved = _state.pop("session")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/predict", json={"text": "Test text", "top_k": 1})
    assert response.status_code == 503
    _state["session"] = saved
