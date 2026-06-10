"""Decoupled OCR-model adapters.

olmOCR hardwires one model: its YAML-front-matter prompt, a rotation-retry
protocol, a guided-decoding regex, and front-matter parsing. paperscale factors
the *model-specific* parts into the small ``OCRModel`` interface below so the
pipeline can drive **any** OpenAI-compatible OCR model that emits Markdown.

A model adapter is responsible for exactly two things:

* :meth:`build_messages` — the chat ``messages`` payload for one page image.
* :meth:`parse` — turn the model's raw text response into a normalized
  :class:`~paperscale.prompts.PageResponse`.

Everything else — page rendering, retry/backoff, validity checks, the pdftotext
fallback, queueing, and output assembly — lives in the pipeline and is identical
across models.
"""

from __future__ import annotations

import abc

from paperscale.prompts import PageResponse


class OCRModel(abc.ABC):
    """Adapter that couples a prompt to a response parser for one OCR model."""

    #: Served-model-name sent in the API request when the user does not pass
    #: ``--model``. For an internal vLLM server this is also the alias the model
    #: is served under.
    default_model_name: str = "paperscale"

    @abc.abstractmethod
    def build_messages(self, image_base64: str) -> list[dict]:
        """Build the chat ``messages`` array for a single page image (PNG, base64)."""

    def guided_regex(self) -> str | None:
        """Regex to constrain decoding when ``--guided-decoding`` is set.

        Returning ``None`` (the default) leaves generation unconstrained, which
        is correct for free-form Markdown models.
        """
        return None

    @abc.abstractmethod
    def parse(self, content: str) -> PageResponse:
        """Parse the model's raw text output into a normalized ``PageResponse``.

        Implementations must not raise on ordinary "empty page" output; return a
        ``PageResponse`` with ``natural_text=None`` instead.
        """
