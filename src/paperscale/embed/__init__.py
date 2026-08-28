"""`paperscale embed` — OCR text to vectors, against an external vLLM server.

Public names only. **No heavy imports at module scope**: numpy, lancedb and
pyarrow ship in the optional ``embed`` extra, and `paperscale.embed` must stay
importable (for `--help`, and for a registry listing) on an OCR-only install.
Every submodule that needs one imports it inside the function that uses it.

The design this implements is `docs/design/embed.md`; its vocabulary is
`CONTEXT.md`'s. Read the design before changing anything here -- most of what
looks arbitrary below is a decision with a measurement behind it.
"""

from __future__ import annotations

__all__ = ["EMBED_MODEL_REGISTRY", "EmbedModel", "build_embed_model"]


def __getattr__(name: str):
    """Lazily forward the three adapter names without importing at module scope.

    `adapters` is pure-Python and cheap, but routing through `__getattr__` keeps
    the promise in the docstring literal: importing this package pulls in
    nothing. A typo then still raises AttributeError at the usual place.
    """
    if name in __all__:
        from paperscale.embed import adapters

        return getattr(adapters, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
