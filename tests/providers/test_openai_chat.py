from __future__ import annotations

import unittest

from tests.harness.fakes import FakeProviderResponse
from tests.harness.imports import require_symbol


class OpenAICompatibleProviderTests(unittest.TestCase):
    def test_adapter_uses_provider_neutral_request_and_fake_client_without_network(self) -> None:
        OpenAICompatibleChatProvider = require_symbol("paperscale.providers.openai_chat", "OpenAICompatibleChatProvider")
        OcrPageRequest = require_symbol("paperscale.providers.base", "OcrPageRequest")

        class FakeChatClient:
            def __init__(self) -> None:
                self.calls = []

            def create_page_ocr(self, request):
                self.calls.append(request)
                return FakeProviderResponse(markdown="# Page\n\nText")

        client = FakeChatClient()
        provider = OpenAICompatibleChatProvider(client=client, endpoint="http://unused.test/v1")
        request = OcrPageRequest(
            page_id="doc:1",
            model="generic-model",
            prompt="Return Markdown only.",
            image_bytes=b"fake-image",
            decoding_options={"temperature": 0.0},
            idempotency_key="fp1",
        )
        response = provider.send(request)
        self.assertEqual(response.markdown, "# Page\n\nText")
        self.assertEqual(client.calls, [request])


if __name__ == "__main__":
    unittest.main()
