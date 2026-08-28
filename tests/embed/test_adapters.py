"""Tests for the embedding-model Adapters.

These are fact tables, so most of what is worth testing is that the facts are the ones the design
recorded and that they are recorded in the *shape* the design chose: a range rather than a list, an
empty string rather than a null, and a rejection rather than a clamp.
"""

import unittest

from paperscale.embed import adapters
from paperscale.embed.adapters import (
    EMBED_MODEL_REGISTRY,
    EmbedModel,
    Nemotron3Embed1B,
    Nemotron3Embed8B,
    Qwen3Embedding0_6B,
    Qwen3Embedding4B,
    Qwen3Embedding8B,
    build_embed_model,
    validate_embed_dim,
)

# The default --embed-dim. 768 has to sit inside every pinned model's range at every size, or the
# default would be an error for some of them.
_DEFAULT_EMBED_DIM = 768


class RegistryTests(unittest.TestCase):
    def test_registry_contents(self):
        self.assertEqual(
            set(EMBED_MODEL_REGISTRY),
            {
                "qwen3-embedding-0.6b",
                "qwen3-embedding-4b",
                "qwen3-embedding-8b",
                "nemotron-3-embed-1b",
                "nemotron-3-embed-8b",
            },
        )

    def test_build_known_models(self):
        self.assertIsInstance(build_embed_model("qwen3-embedding-0.6b"), Qwen3Embedding0_6B)
        self.assertIsInstance(build_embed_model("qwen3-embedding-4b"), Qwen3Embedding4B)
        self.assertIsInstance(build_embed_model("qwen3-embedding-8b"), Qwen3Embedding8B)
        self.assertIsInstance(build_embed_model("nemotron-3-embed-1b"), Nemotron3Embed1B)
        self.assertIsInstance(build_embed_model("nemotron-3-embed-8b"), Nemotron3Embed8B)

    def test_registry_holds_classes_not_instances(self):
        for name, factory in EMBED_MODEL_REGISTRY.items():
            with self.subTest(name=name):
                self.assertTrue(isinstance(factory, type) and issubclass(factory, EmbedModel))

    def test_unknown_model_raises_with_choices(self):
        with self.assertRaises(ValueError) as ctx:
            build_embed_model("nope")
        self.assertIn("--embed-model", str(ctx.exception))
        self.assertIn("qwen3-embedding-8b", str(ctx.exception))
        self.assertIn("nemotron-3-embed-1b", str(ctx.exception))

    def test_there_is_no_default_embed_model(self):
        # Obligation 35's Adapter-side half: --embed-model is required because the registry has
        # nothing to fall back to. Embedding models produce vectors that are meaningless across
        # models, so a default would silently pick the semantics of an entire corpus.
        self.assertFalse(hasattr(adapters, "DEFAULT_EMBED_MODEL"))

    def test_public_names_forward_through_the_package(self):
        # paperscale.embed re-exports these lazily; a rename here has to break loudly there.
        from paperscale import embed

        self.assertIs(embed.EmbedModel, EmbedModel)
        self.assertIs(embed.build_embed_model, build_embed_model)
        self.assertIs(embed.EMBED_MODEL_REGISTRY, EMBED_MODEL_REGISTRY)


class AdapterFactsTests(unittest.TestCase):
    def test_native_dims(self):
        self.assertEqual(build_embed_model("qwen3-embedding-0.6b").native_dim, 1024)
        self.assertEqual(build_embed_model("qwen3-embedding-4b").native_dim, 2560)
        self.assertEqual(build_embed_model("qwen3-embedding-8b").native_dim, 4096)
        self.assertEqual(build_embed_model("nemotron-3-embed-1b").native_dim, 2048)
        self.assertEqual(build_embed_model("nemotron-3-embed-8b").native_dim, 4096)

    def test_native_dim_2560_rules_out_a_powers_of_two_shortcut(self):
        # Recorded as a test because a future "tidy" that stores widths as exponents, or validates
        # against a powers-of-two list, would pass every other assertion in this file.
        native = build_embed_model("qwen3-embedding-4b").native_dim
        self.assertNotEqual(native & (native - 1), 0)

    def test_card_context_length_is_the_card_number_not_the_servers(self):
        # 32768 for both families. A default `vllm serve` advertises 262144 for Nemotron and 40960
        # for Qwen3-8B, i.e. wrong in the direction that silently truncates a Chunk.
        for name in EMBED_MODEL_REGISTRY:
            with self.subTest(name=name):
                self.assertEqual(build_embed_model(name).card_context_length, 32768)

    def test_min_dim_is_the_shared_convention(self):
        self.assertEqual(EmbedModel.min_dim, 32)
        for name in EMBED_MODEL_REGISTRY:
            with self.subTest(name=name):
                self.assertEqual(build_embed_model(name).min_dim, 32)

    def test_default_embed_dim_is_valid_for_every_pinned_model(self):
        for name in EMBED_MODEL_REGISTRY:
            with self.subTest(name=name):
                adapter = build_embed_model(name)
                self.assertEqual(validate_embed_dim(adapter, _DEFAULT_EMBED_DIM), _DEFAULT_EMBED_DIM)

    def test_the_adapter_carries_no_methods(self):
        # An embedding Adapter builds no prompt and parses no response. An abstract method here
        # would mean someone gave it a job that belongs in client.py or vectors.py.
        self.assertEqual(EmbedModel.__abstractmethods__, frozenset())


class InstructionTests(unittest.TestCase):
    def test_qwen3_document_instruction_is_empty_string_never_none(self):
        # "" says *we decided on none*; a missing value says *we did not record it*. Provenance
        # has to keep those apart -- both Sinks write this field, and a Consumer reading null
        # cannot tell an intentional no-prefix from an unrecorded one.
        for name in ("qwen3-embedding-0.6b", "qwen3-embedding-4b", "qwen3-embedding-8b"):
            with self.subTest(name=name):
                instruction = build_embed_model(name).document_instruction
                self.assertIsNotNone(instruction)
                self.assertIsInstance(instruction, str)
                self.assertEqual(instruction, "")

    def test_qwen3_query_instruction_is_a_template_verbatim_from_qwens_helper(self):
        # Space after "Instruct:", none after "Query:" -- character for character, because a
        # Consumer matching the convention gets this string from the Sink and not from the card.
        for name in ("qwen3-embedding-0.6b", "qwen3-embedding-4b", "qwen3-embedding-8b"):
            with self.subTest(name=name):
                self.assertEqual(build_embed_model(name).query_instruction, "Instruct: {task_description}\nQuery:{query}")

    def test_qwen3_query_instruction_renders(self):
        rendered = build_embed_model("qwen3-embedding-8b").query_instruction.format(
            task_description="Given a web search query, retrieve relevant passages",
            query="who signed the lease",
        )
        self.assertEqual(rendered, "Instruct: Given a web search query, retrieve relevant passages\nQuery:who signed the lease")

    def test_nemotron_instructions_are_both_sides_and_both_fixed(self):
        # This family is why the document side is applied at all: the design first generalized
        # from Qwen3 and assumed the convention was query-side only.
        for name in ("nemotron-3-embed-1b", "nemotron-3-embed-8b"):
            with self.subTest(name=name):
                adapter = build_embed_model(name)
                self.assertEqual(adapter.document_instruction, "passage: ")
                self.assertEqual(adapter.query_instruction, "query: ")
                # No slot: the absence of a placeholder is what tells a Consumer to use the
                # string as it stands rather than fill something in.
                self.assertNotIn("{", adapter.query_instruction)


class EmbedDimValidationTests(unittest.TestCase):
    """Obligation 4: outside ``[min_dim, native_dim]`` is rejected; above is rejected, not clamped."""

    def test_both_bounds_are_inclusive(self):
        for name in EMBED_MODEL_REGISTRY:
            with self.subTest(name=name):
                adapter = build_embed_model(name)
                self.assertEqual(validate_embed_dim(adapter, adapter.min_dim), adapter.min_dim)
                self.assertEqual(validate_embed_dim(adapter, adapter.native_dim), adapter.native_dim)

    def test_below_min_dim_is_rejected(self):
        for name in EMBED_MODEL_REGISTRY:
            with self.subTest(name=name):
                adapter = build_embed_model(name)
                with self.assertRaises(SystemExit) as ctx:
                    validate_embed_dim(adapter, adapter.min_dim - 1)
                self.assertIn("--embed-dim", str(ctx.exception))
                self.assertIn(str(adapter.min_dim), str(ctx.exception))

    def test_above_native_dim_is_rejected(self):
        for name in EMBED_MODEL_REGISTRY:
            with self.subTest(name=name):
                adapter = build_embed_model(name)
                with self.assertRaises(SystemExit) as ctx:
                    validate_embed_dim(adapter, adapter.native_dim + 1)
                self.assertIn("--embed-dim", str(ctx.exception))
                self.assertIn(str(adapter.native_dim), str(ctx.exception))

    def test_above_native_dim_is_not_clamped(self):
        # The design's own example: a 2048-wide model asked for 4096. 4096 is a real width in this
        # registry (both 8B models), so the mistake is nearly always "I meant the 8B sibling" --
        # and handing back 2048 would produce a Sink that is indistinguishable from a correct one.
        small = build_embed_model("nemotron-3-embed-1b")
        with self.assertRaises(SystemExit) as ctx:
            validate_embed_dim(small, 4096)
        self.assertIn("4096", str(ctx.exception))
        self.assertIn("2048", str(ctx.exception))

        big = build_embed_model("nemotron-3-embed-8b")
        self.assertEqual(validate_embed_dim(big, 4096), 4096)

    def test_zero_and_negative_are_rejected(self):
        # vLLM's own floor is 1, so nothing below min_dim is caught server-side for us.
        adapter = build_embed_model("qwen3-embedding-8b")
        for value in (0, -1, 1):
            with self.subTest(value=value):
                with self.assertRaises(SystemExit):
                    validate_embed_dim(adapter, value)
