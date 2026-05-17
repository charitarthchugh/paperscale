from __future__ import annotations

import unittest

from paperscale.profiles.builtin import get_builtin_profile
from paperscale.providers.base import PageOcrRequest
from paperscale.providers.openai_chat import OpenAIChatProvider


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    class Responses:
        def __init__(self, outer: "FakeOpenAIClient") -> None:
            self._outer = outer

        def create(self, **kwargs: object) -> object:
            self._outer.calls.append(kwargs)

            class Response:
                output_text = "# Page 1\n\nHello from OCR"

            return Response()

    @property
    def responses(self) -> "FakeOpenAIClient.Responses":
        return FakeOpenAIClient.Responses(self)


class OpenAIChatProviderTests(unittest.TestCase):
    def test_provider_sends_provider_neutral_request_without_real_network(self) -> None:
        profile = get_builtin_profile("generic_vlm_markdown")
        request = profile.build_request(
            page_id="doc:1",
            image_bytes=b"fake-image",
            image_media_type="image/png",
        )
        client = FakeOpenAIClient()
        provider = OpenAIChatProvider(client=client)

        response = provider.send(request)

        self.assertEqual(response.markdown, "# Page 1\n\nHello from OCR")
        self.assertEqual(response.provider_request_id, request.fingerprint)
        self.assertEqual(len(client.calls), 1)
        call = client.calls[0]
        self.assertEqual(call["model"], request.model)
        self.assertEqual(call["temperature"], request.decoding["temperature"])
        self.assertIn("input", call)

    def test_base_request_fingerprint_includes_provider_model_profile_and_image_hash(self) -> None:
        request_a = PageOcrRequest(
            page_id="p1",
            provider="openai-compatible-chat",
            model="model-a",
            profile_name="generic_vlm_markdown",
            profile_version="v1",
            prompt_hash="prompt",
            parser_version="parser-v1",
            image_hash="image-a",
            prompt="Convert to Markdown",
            image_media_type="image/png",
            image_bytes=b"a",
            decoding={"temperature": 0.0},
            render_options={"target_longest_side": 1280},
        )
        request_b = PageOcrRequest(
            page_id="p1",
            provider="openai-compatible-chat",
            model="model-a",
            profile_name="generic_vlm_markdown",
            profile_version="v1",
            prompt_hash="prompt",
            parser_version="parser-v1",
            image_hash="image-b",
            prompt="Convert to Markdown",
            image_media_type="image/png",
            image_bytes=b"b",
            decoding={"temperature": 0.0},
            render_options={"target_longest_side": 1280},
        )

        self.assertNotEqual(request_a.fingerprint, request_b.fingerprint)


if __name__ == "__main__":
    unittest.main()
