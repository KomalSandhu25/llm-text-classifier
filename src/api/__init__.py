"""
llm-text-classifier · API package

Exports the FastAPI application instance so it can be referenced by
uvicorn / gunicorn as ``src.api:app``.
"""

from src.api.main import app  # noqa: F401

__all__ = ["app"]
