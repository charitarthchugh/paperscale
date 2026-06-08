"""Deterministic OpenAI/vLLM-compatible mock inference API for local tests."""

from __future__ import annotations

from typing import Any

__all__ = ["MockApiConfig", "MockApiState", "create_app"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from paperscale.mock_api import app

        return getattr(app, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
