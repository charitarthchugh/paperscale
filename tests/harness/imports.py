"""Import helpers that make TDD red failures actionable."""

from __future__ import annotations

import importlib
from typing import Any


def require_module(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:  # pragma: no cover - failure text is the contract
        raise AssertionError(
            f"Missing production module {module_name!r}; implement the approved VLM OCR v1 API."
        ) from exc


def require_symbol(module_name: str, symbol_name: str) -> Any:
    module = require_module(module_name)
    try:
        return getattr(module, symbol_name)
    except AttributeError as exc:  # pragma: no cover - failure text is the contract
        raise AssertionError(
            f"Missing symbol {module_name}.{symbol_name}; tests encode the approved v1 contract."
        ) from exc
