"""Tests for the MRL slice, the re-normalization, and the token-weighted pool.

Numbers here are generated from a seeded `default_rng`, not hand-written: the
properties under test (unit length, recomputability, bits that do or do not move)
only mean something over vectors wide enough to round like the real ones, and 768
hand-written floats would be worse than useless.
"""

import dataclasses
import datetime
import unittest

import numpy as np

from paperscale.embed.vectors import EmbeddedDocument, pool_document_vector, slice_and_normalize

_NATIVE_DIM = 1024
_STORED_DIM = 768


def _server_vectors(n: int, dim: int = _NATIVE_DIM, seed: int = 0) -> np.ndarray:
    """`n` unit-length float32 rows -- the shape and normalization vLLM returns."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n, dim)).astype(np.float32)
    return (raw / np.linalg.norm(raw, axis=1)[:, None]).astype(np.float32)


class SliceAndNormalizeTest(unittest.TestCase):
    def test_shape_and_dtype(self):
        out = slice_and_normalize(_server_vectors(3), _STORED_DIM)
        self.assertEqual(out.shape, (3, _STORED_DIM))
        self.assertEqual(out.dtype, np.float32)

    def test_every_row_is_unit_length(self):
        out = slice_and_normalize(_server_vectors(8), _STORED_DIM)
        for i, norm in enumerate(np.linalg.norm(out.astype(np.float64), axis=1)):
            self.assertAlmostEqual(norm, 1.0, places=6, msg=f"row {i}")

    def test_slicing_client_side_matches_asking_the_server(self):
        # The whole justification for never sending `dimensions`: the server would
        # compute normalize(slice(raw)) on the un-normalized pooled output, and
        # paperscale computes normalize(slice(normalize(raw))). The positive scalar
        # cancels, so the two are the same unit vector to float rounding.
        rng = np.random.default_rng(7)
        raw = rng.standard_normal((4, _NATIVE_DIM)).astype(np.float32)
        server_side = raw[:, :_STORED_DIM].astype(np.float64)
        server_side = server_side / np.linalg.norm(server_side, axis=1)[:, None]

        native_unit = (raw / np.linalg.norm(raw, axis=1)[:, None]).astype(np.float32)
        client_side = slice_and_normalize(native_unit, _STORED_DIM)

        self.assertTrue(np.allclose(client_side, server_side, rtol=0, atol=1e-6))

    def test_zero_chunk_batch_keeps_the_stored_width(self):
        out = slice_and_normalize(np.zeros((0, _NATIVE_DIM), dtype=np.float32), _STORED_DIM)
        self.assertEqual(out.shape, (0, _STORED_DIM))
        self.assertEqual(out.dtype, np.float32)

    def test_response_narrower_than_stored_dim_is_rejected(self):
        # numpy would slice a 512-wide response to 512 without a word, and every
        # later shape check would agree with it.
        with self.assertRaises(ValueError) as caught:
            slice_and_normalize(_server_vectors(2, dim=512), _STORED_DIM)
        self.assertIn("512", str(caught.exception))
        self.assertIn(str(_STORED_DIM), str(caught.exception))

    def test_zero_norm_row_is_an_error_never_nan(self):
        vectors = _server_vectors(3)
        vectors[1, :_STORED_DIM] = 0.0
        with self.assertRaises(ValueError) as caught:
            slice_and_normalize(vectors, _STORED_DIM)
        self.assertIn("1", str(caught.exception))

    def test_non_finite_row_is_an_error_never_nan(self):
        vectors = _server_vectors(2)
        vectors[0, 0] = np.float32("nan")
        with self.assertRaises(ValueError):
            slice_and_normalize(vectors, _STORED_DIM)

    def test_an_infinite_row_is_an_error_never_nan(self):
        # `inf` is not a second flavour of the case above, it is the input that *makes* one:
        # its norm is `inf`, which passes a `> 0` guard cleanly, and `inf / inf` is NaN. The
        # row above arrives as NaN and is caught; this one arrives looking finite and leaves
        # as `[nan, 0, 0, ...]`, which the `.npz` Sink shape-casts and stores.
        #
        # The row number is asserted because `argmin` picks the wrong one here -- against an
        # infinite norm it returns the smallest *finite* entry, naming a row that is fine.
        for coordinate in (np.float32("inf"), np.float32("-inf")):
            with self.subTest(coordinate=coordinate):
                vectors = _server_vectors(2)
                vectors[1, 0] = coordinate
                with self.assertRaises(ValueError) as caught:
                    slice_and_normalize(vectors, _STORED_DIM)
                self.assertIn("vector 1", str(caught.exception))


class SingleChunkTest(unittest.TestCase):
    """The identity case, which the design requires to be exact by construction."""

    def test_document_vector_is_bit_identical_to_the_chunk_vector(self):
        chunk_vectors = slice_and_normalize(_server_vectors(1, seed=3), _STORED_DIM)
        doc = pool_document_vector(chunk_vectors, [17_000])

        self.assertEqual(doc.dtype, chunk_vectors.dtype)
        self.assertEqual(doc.tobytes(), chunk_vectors[0].tobytes())
        self.assertTrue(np.array_equal(doc, chunk_vectors[0]), "the single-Chunk Document vector must be a copy, not a re-normalized mean")

    def test_the_copy_does_not_alias_the_chunk_vector(self):
        chunk_vectors = slice_and_normalize(_server_vectors(1, seed=4), _STORED_DIM)
        doc = pool_document_vector(chunk_vectors, [1])
        doc[0] = 0.5
        self.assertNotEqual(float(chunk_vectors[0, 0]), 0.5)

    def test_renormalizing_a_single_vector_really_does_move_the_bits(self):
        # Why the short-circuit exists at all: the divisor is 1 +/- eps, not 1.
        # Measured at numpy 2.5.2, all 64 of these come back bitwise different;
        # asserting "at least one" keeps the test from tracking a libm change.
        chunk_vectors = slice_and_normalize(_server_vectors(64, seed=5), _STORED_DIM)
        moved = 0
        for row in chunk_vectors:
            weighted = np.float32(1234.0) * row
            renormalized = (weighted / np.linalg.norm(weighted)).astype(np.float32)
            if not np.array_equal(renormalized, row):
                moved += 1
        self.assertGreater(moved, 0)


class PoolDocumentVectorTest(unittest.TestCase):
    def test_recomputable_from_the_stored_arrays(self):
        # The property a Consumer holding one Sink depends on: chunk_vectors and
        # token_count are all it needs to rebuild the Document vector.
        chunk_vectors = slice_and_normalize(_server_vectors(5, seed=11), _STORED_DIM)
        token_counts = [12_000, 9_000, 31_000, 700, 5]
        doc = pool_document_vector(chunk_vectors, token_counts)

        weights = np.asarray(token_counts, dtype=np.int32)
        pooled = (chunk_vectors * weights[:, None]).sum(axis=0)
        expected = (pooled / np.linalg.norm(pooled)).astype(np.float32)

        self.assertTrue(np.allclose(doc, expected, rtol=0, atol=1e-6))

    def test_document_vector_is_unit_length(self):
        chunk_vectors = slice_and_normalize(_server_vectors(6, seed=12), _STORED_DIM)
        doc = pool_document_vector(chunk_vectors, [31_000, 28_000, 30_500, 2, 900, 17])
        self.assertEqual(doc.shape, (_STORED_DIM,))
        self.assertEqual(doc.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(doc.astype(np.float64))), 1.0, places=6)

    def test_the_one_page_tail_does_not_get_half_the_document(self):
        # Greedy packing's worst case, made small: forty pages of Chunk plus a
        # one-page tail. A uniform mean sits at 45 degrees between them.
        body = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        tail = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        chunk_vectors = np.stack([body, tail])

        weighted = pool_document_vector(chunk_vectors, [32_000, 800])
        uniform = pool_document_vector(chunk_vectors, [1, 1])

        self.assertGreater(float(weighted @ body), 0.99)
        self.assertAlmostEqual(float(uniform @ body), 0.5**0.5, places=6)

    def test_zero_total_weight_falls_back_to_uniform(self):
        chunk_vectors = slice_and_normalize(_server_vectors(3, seed=13), _STORED_DIM)
        fallback = pool_document_vector(chunk_vectors, [0, 0, 0])
        explicit = pool_document_vector(chunk_vectors, [1, 1, 1])
        self.assertTrue(np.array_equal(fallback, explicit))

    def test_cancelling_chunks_are_an_error_never_nan(self):
        row = slice_and_normalize(_server_vectors(1, seed=14), _STORED_DIM)[0]
        chunk_vectors = np.stack([row, -row])
        with self.assertRaises(ValueError):
            pool_document_vector(chunk_vectors, [1_000, 1_000])

    def test_no_chunks_gives_an_empty_document_vector(self):
        doc = pool_document_vector(np.zeros((0, _STORED_DIM), dtype=np.float32), [])
        self.assertEqual(doc.shape, (0,))
        self.assertEqual(doc.size, 0)
        self.assertEqual(doc.dtype, np.float32)

    def test_weights_must_be_parallel_to_the_chunk_vectors(self):
        chunk_vectors = slice_and_normalize(_server_vectors(3, seed=15), _STORED_DIM)
        with self.assertRaises(ValueError):
            pool_document_vector(chunk_vectors, [10, 20])


class EmbeddedDocumentTest(unittest.TestCase):
    def _build(self) -> EmbeddedDocument:
        chunk_vectors = slice_and_normalize(_server_vectors(2, seed=21), _STORED_DIM)
        return EmbeddedDocument(
            document_name="law/pdfs/doc9419897.pdf",
            run_label="nemotron-8b",
            source_file="/run/media/cc/data/law/pdfs/doc9419897.pdf",
            source_digest="9f2b1c4d8e0a3f57",
            created=datetime.datetime(2026, 8, 18, 9, 4, 37, tzinfo=datetime.UTC),
            chunks=[],
            chunk_vectors=chunk_vectors,
            document_vector=pool_document_vector(chunk_vectors, [30_000, 400]),
        )

    def test_created_is_timezone_aware_utc(self):
        # Both Sinks serialize it -- ISO-8601 Z in the sidecar, timestamp[us, UTC]
        # in LanceDB -- and a naive datetime means "the writing machine's clock".
        created = self._build().created
        self.assertIsNotNone(created.tzinfo)
        self.assertEqual(created.utcoffset(), datetime.timedelta(0))

    def test_is_frozen(self):
        doc = self._build()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            doc.run_label = "other"  # type: ignore[misc]

    def test_carries_the_raw_source_file_not_the_derived_name(self):
        doc = self._build()
        self.assertTrue(doc.source_file.startswith("/"))
        self.assertFalse(doc.document_name.startswith("/"))


if __name__ == "__main__":
    unittest.main()
