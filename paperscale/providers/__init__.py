"""Provider adapters and provider-neutral OCR request contracts."""

from paperscale.providers.base import PageOcrRequest, PageOcrResponse, ProviderError

__all__ = ["PageOcrRequest", "PageOcrResponse", "ProviderError"]
