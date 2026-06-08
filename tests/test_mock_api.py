from __future__ import annotations

import base64
import contextlib
import io
import json
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from openai import OpenAI

from paperscale.mock_api import MockApiConfig, MockApiState, create_app
from paperscale.profiles.builtin import get_builtin_profile
from paperscale.providers.openai_chat import OpenAIChatProvider


def _data_url(image: bytes = b"page-image", media_type: str = "image/png") -> str:
    return f"data:{media_type};base64,{base64.b64encode(image).decode('ascii')}"


def _responses_payload(*, model: str = "mock-vlm", image: bytes = b"page-image") -> dict[str, Any]:
    return {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Convert page 7 to Markdown"},
                    {"type": "input_image", "image_url": _data_url(image)},
                ],
            }
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 128,
    }


def _chat_payload(*, model: str = "mock-vlm", image: bytes = b"chat-image") -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "OCR this page"},
                    {"type": "image_url", "image_url": {"url": _data_url(image, "image/jpeg")}},
                ],
            }
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 128,
    }


class MockApiEndpointTests(unittest.TestCase):
    def test_models_lists_served_model(self) -> None:
        client = TestClient(create_app(MockApiConfig(served_model="mock-vlm")))

        response = client.get("/v1/models")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["object"], "list")
        self.assertEqual(payload["data"][0]["id"], "mock-vlm")

    def test_responses_endpoint_accepts_current_provider_request_shape(self) -> None:
        state = MockApiState()
        client = TestClient(create_app(MockApiConfig(served_model="mock-vlm"), state=state))

        response = client.post("/v1/responses", json=_responses_payload(image=b"abc"))

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["model"], "mock-vlm")
        self.assertEqual(payload["object"], "response")
        self.assertIn("endpoint: responses", payload["output_text"])
        self.assertIn("image_sha256: ba7816bf8f01cfea", payload["output_text"])
        self.assertIn("prompt_sha256:", payload["output_text"])
        self.assertEqual(len(state.requests), 1)
        self.assertEqual(state.requests[0]["endpoint"], "responses")
        self.assertEqual(state.requests[0]["image_sha256"], "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

    def test_chat_completions_endpoint_accepts_vllm_multimodal_message_shape(self) -> None:
        client = TestClient(create_app(MockApiConfig(served_model="mock-vlm")))

        response = client.post("/v1/chat/completions", json=_chat_payload(image=b"xyz"))

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        self.assertEqual(payload["object"], "chat.completion")
        self.assertIn("endpoint: chat.completions", content)
        self.assertIn("model: mock-vlm", content)
        self.assertIn("image_sha256: 3608bca1e44ea6c4", content)

    def test_validation_errors_are_openai_style(self) -> None:
        cases = [
            ("wrong model", _responses_payload(model="missing-model"), 404, "model_not_found"),
            ("missing image", {"model": "mock-vlm", "input": [{"role": "user", "content": [{"type": "input_text", "text": "only text"}]}]}, 400, "missing_image"),
            ("bad data url", {**_responses_payload(), "input": [{"role": "user", "content": [{"type": "input_image", "image_url": "not-a-data-url"}]}]}, 400, "invalid_image_url"),
            ("bad base64", {**_responses_payload(), "input": [{"role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64,%%%"}]}]}, 400, "invalid_image_base64"),
            ("oversize image", _responses_payload(image=b"12345"), 413, "image_too_large"),
            ("bad temperature", {**_responses_payload(), "temperature": 3.0}, 400, "invalid_temperature"),
            ("bad top_p", {**_responses_payload(), "top_p": 0.0}, 400, "invalid_top_p"),
            ("bad tokens", {**_responses_payload(), "max_tokens": 0}, 400, "invalid_max_tokens"),
        ]
        client = TestClient(create_app(MockApiConfig(served_model="mock-vlm", max_image_bytes=3)))

        for name, payload, status, code in cases:
            with self.subTest(name=name):
                response = client.post("/v1/responses", json=payload)
                self.assertEqual(response.status_code, status)
                error = response.json()["error"]
                self.assertEqual(error["code"], code)
                self.assertIn("message", error)
                self.assertIn("type", error)

    def test_bearer_auth_failure_uses_openai_style_401(self) -> None:
        client = TestClient(create_app(MockApiConfig(bearer_token="secret")))

        response = client.post("/v1/responses", json=_responses_payload())

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers["www-authenticate"], "Bearer")
        self.assertEqual(response.json()["error"]["code"], "invalid_api_key")

        ok = client.post(
            "/v1/responses",
            json=_responses_payload(),
            headers={"Authorization": "Bearer secret"},
        )
        self.assertEqual(ok.status_code, 200)

    def test_scenarios_return_outputs_that_exercise_profile_parsers(self) -> None:
        state = MockApiState(scenario="empty_output")
        client = TestClient(create_app(MockApiConfig(), state=state))
        profile = get_builtin_profile("generic_vlm_markdown")

        empty = client.post("/v1/responses", json=_responses_payload()).json()["output_text"]
        self.assertFalse(profile.parse_and_validate(empty).ok)

        client.post("/__paperscale/scenario", json={"scenario": "repeated_ngram"})
        repeated = client.post("/v1/responses", json=_responses_payload()).json()["output_text"]
        repeated_result = profile.parse_and_validate(repeated)
        self.assertFalse(repeated_result.ok)
        self.assertEqual(repeated_result.retry_classification, "retryable")

        client.post("/__paperscale/scenario", json={"scenario": "json_layout"})
        layout = client.post("/v1/responses", json=_responses_payload()).json()["output_text"]
        layout_result = get_builtin_profile("glm_ocr").parse_and_validate(layout)
        self.assertTrue(layout_result.ok)
        self.assertIn("regions", layout_result.metadata)

    def test_rate_limit_then_ok_and_overload_return_bounded_429(self) -> None:
        state = MockApiState(scenario="rate_limit_then_ok")
        client = TestClient(create_app(MockApiConfig(), state=state))

        first = client.post("/v1/responses", json=_responses_payload())
        second = client.post("/v1/responses", json=_responses_payload())

        self.assertEqual(first.status_code, 429)
        self.assertEqual(first.json()["error"]["code"], "rate_limit_exceeded")
        self.assertEqual(second.status_code, 200)

        overloaded = TestClient(create_app(MockApiConfig(max_in_flight=0)))
        response = overloaded.post("/v1/responses", json=_responses_payload())
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.json()["error"]["code"], "server_overloaded")

    def test_debug_request_log_and_reset(self) -> None:
        client = TestClient(create_app(MockApiConfig()))
        client.post("/v1/responses", json=_responses_payload(image=b"one"))

        logged = client.get("/__paperscale/requests")
        self.assertEqual(logged.status_code, 200)
        self.assertEqual(logged.json()["requests"][0]["endpoint"], "responses")

        reset = client.post("/__paperscale/reset")
        self.assertEqual(reset.status_code, 200)
        self.assertEqual(client.get("/__paperscale/requests").json()["requests"], [])


class MockApiLocalhostSmokeTests(unittest.TestCase):

    def test_console_entry_module_imports_without_mock_api_extra_dependencies(self) -> None:
        code = r"""
import importlib.abc
import sys

class BlockMockApiDeps(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'fastapi' or fullname.startswith('uvicorn'):
            raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, BlockMockApiDeps())
import paperscale.mock_api.cli
print('ok')
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "ok")

    def test_cli_server_handles_openai_sdk_provider_request_and_records_it(self) -> None:
        port = _free_port()
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "paperscale.mock_api.cli",
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--model",
                "mock-vlm",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            _wait_for_models(port)
            profile = get_builtin_profile("generic_vlm_markdown")
            request = profile.build_request("doc-1:page-1", b"real-page-image", "image/png", model="mock-vlm")
            provider = OpenAIChatProvider(
                client=OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="test-key")
            )

            response = provider.send(request)

            self.assertIn("model: mock-vlm", response.markdown)
            self.assertIn(f"image_sha256: {request.image_hash[:16]}", response.markdown)
            self.assertEqual(response.metadata["model"], "mock-vlm")
            log_payload = _get_json(f"http://127.0.0.1:{port}/__paperscale/requests")
            self.assertEqual(len(log_payload["requests"]), 1)
            self.assertEqual(log_payload["requests"][0]["image_sha256"], request.image_hash)
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            stderr = process.stderr.read() if process.stderr else ""
            if process.stderr:
                process.stderr.close()
            if process.returncode not in (0, -15):
                self.fail(f"mock API server exited unexpectedly with {process.returncode}: {stderr}")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_models(port: int) -> None:
    deadline = time.monotonic() + 10
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.monotonic() < deadline:
        try:
            _get_json(url)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(0.1)
    raise AssertionError("mock API server did not become ready")


def _get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=2) as response:  # noqa: S310 - localhost test URL
        return json.loads(response.read().decode("utf-8"))


if __name__ == "__main__":
    unittest.main()

class MockApiCoverageExpansionTests(unittest.TestCase):
    def test_main_application_profile_provider_and_assembly_flow_uses_mock_api_request_log(self) -> None:
        port = _free_port()
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "paperscale.mock_api.cli",
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--model",
                "mock-vlm",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            _wait_for_models(port)
            profile = get_builtin_profile("generic_vlm_markdown")
            provider = OpenAIChatProvider(
                client=OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="test-key")
            )
            page_payloads: list[dict[str, object]] = []
            expected_hashes: list[str] = []
            for page_number, image in [(1, b"page-one-image"), (2, b"page-two-image")]:
                request = profile.build_request(
                    f"doc-mock:page-{page_number}", image, "image/png", model="mock-vlm"
                )
                expected_hashes.append(request.image_hash)
                provider_response = provider.send(request)
                parsed = profile.parse_and_validate(provider_response.markdown)
                self.assertTrue(parsed.ok, parsed.diagnostic)
                page_payloads.append(
                    {
                        "document_id": "doc-mock",
                        "page_number": page_number,
                        "markdown": parsed.markdown,
                    }
                )

            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                input_path = tmp_path / "pages.jsonl"
                output_path = tmp_path / "assembled.md"
                input_path.write_text(
                    "".join(json.dumps(payload) + "\n" for payload in page_payloads),
                    encoding="utf-8",
                )

                from paperscale.cli import main as paperscale_main

                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    exit_code = paperscale_main(
                        ["assemble", "--input", str(input_path), "--output", str(output_path), "--title", "Mock Doc"]
                    )

                assembled = output_path.read_text(encoding="utf-8")
                self.assertEqual(exit_code, 0)
                self.assertIn("# Mock Doc", assembled)
                self.assertIn("<!-- page-break -->", assembled)
                for image_hash in expected_hashes:
                    self.assertIn(f"image_sha256: {image_hash[:16]}", assembled)

            log_payload = _get_json(f"http://127.0.0.1:{port}/__paperscale/requests")
            self.assertEqual([entry["image_sha256"] for entry in log_payload["requests"]], expected_hashes)
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            stderr = process.stderr.read() if process.stderr else ""
            if process.stderr:
                process.stderr.close()
            if process.returncode not in (0, -15):
                self.fail(f"mock API server exited unexpectedly with {process.returncode}: {stderr}")

    def test_config_and_state_reject_invalid_settings(self) -> None:
        for kwargs, message in [
            ({"served_model": ""}, "served_model"),
            ({"scenario": "unknown"}, "scenario"),
            ({"max_image_bytes": -1}, "max_image_bytes"),
            ({"max_in_flight": -1}, "max_in_flight"),
            ({"latency_ms": -1}, "latency_ms"),
        ]:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    MockApiConfig(**kwargs)
        with self.assertRaisesRegex(ValueError, "scenario"):
            MockApiState(scenario="unknown")

    def test_auth_applies_to_model_and_debug_endpoints(self) -> None:
        client = TestClient(create_app(MockApiConfig(bearer_token="secret")))
        for method, path in [(client.get, "/v1/models"), (client.get, "/__paperscale/requests")]:
            with self.subTest(path=path):
                response = method(path)
                self.assertEqual(response.status_code, 401)
                self.assertEqual(response.json()["error"]["code"], "invalid_api_key")
        response = client.post("/__paperscale/reset")
        self.assertEqual(response.status_code, 401)
        response = client.post("/__paperscale/scenario", json={"scenario": "ok_markdown"})
        self.assertEqual(response.status_code, 401)

    def test_invalid_json_and_shape_errors_are_openai_style(self) -> None:
        client = TestClient(create_app(MockApiConfig()))
        invalid_json = client.post("/v1/responses", content="{", headers={"content-type": "application/json"})
        self.assertEqual(invalid_json.status_code, 400)
        self.assertEqual(invalid_json.json()["error"]["code"], "invalid_json")

        not_object = client.post("/v1/responses", json=["not", "object"])
        self.assertEqual(not_object.status_code, 400)
        self.assertEqual(not_object.json()["error"]["code"], "invalid_json")

    def test_additional_validation_branches_for_responses_and_chat(self) -> None:
        client = TestClient(create_app(MockApiConfig()))
        cases = [
            ("missing model", "/v1/responses", {"input": []}, "missing_model"),
            ("invalid input", "/v1/responses", {"model": "mock-vlm", "input": "bad"}, "invalid_input"),
            (
                "unsupported media",
                "/v1/responses",
                {
                    **_responses_payload(),
                    "input": [
                        {
                            "role": "user",
                            "content": [{"type": "input_image", "image_url": _data_url(b"gif", "image/gif")}],
                        }
                    ],
                },
                "unsupported_image_media_type",
            ),
            (
                "invalid chat messages",
                "/v1/chat/completions",
                {"model": "mock-vlm", "messages": "bad"},
                "invalid_messages",
            ),
            (
                "missing chat image",
                "/v1/chat/completions",
                {"model": "mock-vlm", "messages": [{"role": "user", "content": "just text"}]},
                "missing_image",
            ),
            (
                "bad max output tokens",
                "/v1/responses",
                {**_responses_payload(), "max_output_tokens": 0},
                "invalid_max_output_tokens",
            ),
        ]
        for name, path, payload, code in cases:
            with self.subTest(name=name):
                response = client.post(path, json=payload)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json()["error"]["code"], code)

    def test_chat_accepts_string_image_url_and_skips_malformed_parts(self) -> None:
        client = TestClient(create_app(MockApiConfig()))
        payload = {
            "model": "mock-vlm",
            "messages": [
                "ignored",
                {"role": "user", "content": [{"type": "text", "text": "OCR"}, "ignored", {"type": "image_url", "image_url": _data_url(b"jpeg", "image/jpeg")}]},
            ],
        }
        response = client.post("/v1/chat/completions", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn("endpoint: chat.completions", response.json()["choices"][0]["message"]["content"])

    def test_remaining_scenarios_return_expected_status_or_payload_shape(self) -> None:
        client = TestClient(create_app(MockApiConfig()))
        for scenario in ["refusal", "malformed_frontmatter", "truncated", "slow"]:
            with self.subTest(scenario=scenario):
                self.assertEqual(client.post("/__paperscale/scenario", json={"scenario": scenario}).status_code, 200)
                response = client.post("/v1/responses", json=_responses_payload())
                self.assertEqual(response.status_code, 200)
                self.assertIsInstance(response.json()["output_text"], str)
        self.assertEqual(client.post("/__paperscale/scenario", json={"scenario": "unknown"}).json()["error"]["code"], "invalid_scenario")
        self.assertEqual(client.post("/__paperscale/scenario", json={"scenario": 3}).json()["error"]["code"], "invalid_scenario")

        self.assertEqual(client.post("/__paperscale/scenario", json={"scenario": "rate_limit"}).status_code, 200)
        rate_limited = client.post("/v1/responses", json=_responses_payload())
        self.assertEqual(rate_limited.status_code, 429)
        self.assertEqual(rate_limited.json()["error"]["code"], "rate_limit_exceeded")

        self.assertEqual(client.post("/__paperscale/scenario", json={"scenario": "server_error"}).status_code, 200)
        errored = client.post("/v1/responses", json=_responses_payload())
        self.assertEqual(errored.status_code, 500)
        self.assertEqual(errored.json()["error"]["code"], "server_error")

    def test_mock_api_cli_calls_uvicorn_with_created_app(self) -> None:
        from unittest.mock import patch

        from paperscale.mock_api import cli

        calls: list[dict[str, object]] = []

        def fake_run(app: object, *, host: str, port: int) -> None:
            calls.append({"app": app, "host": host, "port": port})

        with patch("uvicorn.run", side_effect=fake_run):
            exit_code = cli.main(
                [
                    "serve",
                    "--host",
                    "127.0.0.2",
                    "--port",
                    "8010",
                    "--model",
                    "mock-vlm",
                    "--scenario",
                    "ok_markdown",
                    "--max-image-bytes",
                    "99",
                    "--max-in-flight",
                    "1",
                    "--latency-ms",
                    "1",
                    "--bearer-token",
                    "secret",
                ]
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(calls[0]["host"], "127.0.0.2")
        self.assertEqual(calls[0]["port"], 8010)

    def test_mock_api_package_lazy_exports_and_unknown_attribute(self) -> None:
        import paperscale.mock_api as mock_api

        self.assertIs(mock_api.MockApiConfig, MockApiConfig)
        with self.assertRaises(AttributeError):
            getattr(mock_api, "not_exported")
