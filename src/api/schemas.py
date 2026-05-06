"""
Pydantic request / response schemas for the /predict and /health endpoints.

All field descriptions are included so that the auto-generated OpenAPI docs
(available at /docs) are fully self-documenting.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Payload accepted by POST /predict."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=8_192,
        description="Raw input text to classify.",
        examples=["BREAKING: Scientists discover new exoplanet in habitable zone."],
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of top labels to return, ordered by descending score.",
    )


class PredictResponse(BaseModel):
    """Response returned by POST /predict."""

    labels: List[str] = Field(
        ...,
        description="Top-k predicted class labels, ordered by descending score.",
    )
    scores: List[float] = Field(
        ...,
        description="Softmax probability for each label (sums to <= 1 over top_k).",
    )
    latency_ms: float = Field(
        ...,
        description="Total server-side inference latency in milliseconds.",
    )


class HealthResponse(BaseModel):
    """Response returned by GET /health."""

    status: str = Field(default="ok", description="Service liveness indicator.")
    model_loaded: bool = Field(
        ..., description="Whether the ONNX inference engine is loaded and ready."
    )
    device: str = Field(
        ..., description="Execution provider in use (e.g. CPUExecutionProvider)."
    )
