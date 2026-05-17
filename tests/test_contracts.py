from __future__ import annotations

import unittest

from tests.harness.imports import require_symbol


class ContractsSchemaTests(unittest.TestCase):
    def test_page_attempt_states_match_crash_recovery_model(self) -> None:
        PageAttemptState = require_symbol("paperscale.contracts", "PageAttemptState")
        self.assertEqual(
            {state.value for state in PageAttemptState},
            {
                "pending",
                "reserved",
                "in_flight",
                "succeeded",
                "failed_retryable",
                "failed_terminal",
                "ambiguous",
                "superseded",
            },
        )

    def test_unknown_future_schema_version_fails_closed_without_mutation(self) -> None:
        UnknownSchemaVersionError = require_symbol("paperscale.contracts", "UnknownSchemaVersionError")
        load_versioned_record = require_symbol("paperscale.contracts", "load_versioned_record")
        mutations: list[str] = []
        with self.assertRaises(UnknownSchemaVersionError):
            load_versioned_record(
                {"schema_version": 999, "kind": "page_attempt", "attempt_id": "a1"},
                expected_kind="page_attempt",
                current_version=1,
                on_mutation=lambda *_: mutations.append("mutated"),
            )
        self.assertEqual(mutations, [], "unknown schemas must not mutate or best-effort downgrade")

    def test_request_fingerprint_changes_for_profile_provider_render_and_image_inputs(self) -> None:
        build_provider_request_fingerprint = require_symbol(
            "paperscale.contracts", "build_provider_request_fingerprint"
        )
        base = {
            "provider": "self-hosted-openai-compatible",
            "provider_profile_fingerprint": "provider-capacity-v1",
            "model": "deepseek-ai/DeepSeek-OCR-2",
            "model_profile": "deepseek_ocr_2",
            "model_profile_version": "v1",
            "prompt_hash": "prompt-a",
            "parser_schema_version": 1,
            "decoding_options": {"temperature": 0.0, "max_tokens": 4096},
            "render_options": {"longest_side": 1280, "crop_mode": "dynamic"},
            "page_image_hash": "sha256:image-a",
        }
        original = build_provider_request_fingerprint(**base)
        for field, changed in {
            "provider": "remote-openai-compatible",
            "provider_profile_fingerprint": "provider-capacity-v2",
            "model_profile_version": "v2",
            "prompt_hash": "prompt-b",
            "parser_schema_version": 2,
            "decoding_options": {"temperature": 0.1, "max_tokens": 4096},
            "render_options": {"longest_side": 1024, "crop_mode": "dynamic"},
            "page_image_hash": "sha256:image-b",
        }.items():
            candidate = dict(base)
            candidate[field] = changed
            self.assertNotEqual(
                original,
                build_provider_request_fingerprint(**candidate),
                f"changing {field} must create a distinct request key",
            )


if __name__ == "__main__":
    unittest.main()
