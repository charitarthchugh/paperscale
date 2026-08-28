"""Tests for the Document name -- design 7, test obligations 12-17 and 22.

The name is the ``.npz`` filename, the LanceDB primary key and the Resume key at once, so
these tests are less about strings than about the three failures a wrong string causes.
Two of them pin a *divergence* from ``pipeline.get_markdown_path`` rather than agreement
with it, and say why in place: an assertion that only records today's output would be
silently "fixed" back to the export's behaviour by the next reader.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path

from paperscale.embed import names


class ExtensionAppendTest(unittest.TestCase):
    """Obligation 12 -- the extension is appended, never replaced (design 7.1 rule 3)."""

    def test_extension_is_kept_in_the_name(self):
        self.assertEqual(names.document_name("case.pdf"), "case.pdf")
        self.assertEqual(names.document_name("case.tiff"), "case.tiff")

    def test_siblings_differing_only_by_extension_get_distinct_outputs(self):
        pdf = names.document_name("law/case.pdf")
        tiff = names.document_name("law/case.tiff")
        self.assertEqual(pdf + ".npz", "law/case.pdf.npz")
        self.assertEqual(tiff + ".npz", "law/case.tiff.npz")
        self.assertNotEqual(pdf, tiff)

    def test_markdown_export_collapses_the_same_pair(self):
        """Pins issue #32: the precedent is wrong here, which is why embed diverges.

        If this ever starts failing, ``get_markdown_path`` has been fixed and the
        divergence documented in ``names.py`` needs its reasoning revisited -- not the
        appending behaviour, which is correct either way.
        """
        from paperscale.pipeline import get_markdown_path

        self.assertEqual(
            get_markdown_path("ws", "law/case.pdf"),
            get_markdown_path("ws", "law/case.tiff"),
        )


class NormalizationTest(unittest.TestCase):
    """Obligations 13 and 14 -- normpath resolves, the guard strips what it leaves."""

    def test_interior_dotdot_is_resolved_not_dropped(self):
        # The export's sanitizer drops "..", which makes these two the same Document.
        self.assertEqual(names.document_name("/a/b/../c.pdf"), "a/c.pdf")
        self.assertEqual(names.document_name("/a/b/c.pdf"), "a/b/c.pdf")
        self.assertNotEqual(names.document_name("/a/b/../c.pdf"), names.document_name("/a/b/c.pdf"))

    def test_single_dot_component_is_resolved(self):
        # The export keeps ".", so it names a file "a/./b.pdf" that lands at "a/b.pdf".
        self.assertEqual(names.document_name("/a/./b.pdf"), "a/b.pdf")
        self.assertEqual(names.document_name("/a/b.pdf"), "a/b.pdf")

    def test_leading_dotdot_survives_normpath(self):
        """The precondition for keeping the traversal guard after normalization."""
        self.assertTrue(os.path.normpath("../a/b.pdf").startswith(".."))

    def test_leading_dotdot_is_removed_by_the_guard(self):
        self.assertEqual(names.document_name("../a/b.pdf"), "a/b.pdf")
        self.assertEqual(names.document_name("a/../../b.pdf"), "b.pdf")

    def test_root_and_repeated_slashes_are_stripped(self):
        self.assertEqual(names.document_name("/a/b.pdf"), "a/b.pdf")
        # POSIX normpath preserves exactly two leading slashes; lstrip has to survive it.
        self.assertEqual(names.document_name("//a//b.pdf"), "a/b.pdf")

    def test_name_never_escapes_the_output_directory(self):
        for source in ("../../../etc/passwd.pdf", "/../a.pdf", "a/../../../b.pdf"):
            with self.subTest(source=source):
                name = names.document_name(source)
                self.assertFalse(name.startswith("/"))
                self.assertNotIn("..", name.split("/"))


class TarballFormTest(unittest.TestCase):
    """Design 7.1 rule 1 -- the archive becomes a directory."""

    def test_archive_basename_becomes_a_directory(self):
        self.assertEqual(names.document_name("corpus.tar.gz::internal/doc.pdf"), "corpus/internal/doc.pdf")
        self.assertEqual(names.document_name("/data/corpus.tgz::doc.pdf"), "corpus/doc.pdf")

    def test_internal_path_cannot_escape_the_archive_directory(self):
        self.assertEqual(names.document_name("corpus.tar.gz::../../doc.pdf"), "doc.pdf")


class DigestTest(unittest.TestCase):
    """Obligation 15 -- sha256 over the Source-File string, never over the text."""

    def test_matches_sha256_of_the_raw_string(self):
        source = "/media/cc/data/law/doc9419897.pdf"
        self.assertEqual(names.source_digest(source), hashlib.sha256(source.encode()).hexdigest()[:16])

    def test_is_sixteen_hex_characters(self):
        digest = names.source_digest("/a/b.pdf")
        self.assertEqual(len(digest), 16)
        self.assertEqual(digest, digest.lower())
        self.assertTrue(all(c in "0123456789abcdef" for c in digest))

    def test_survives_a_re_ocr_of_the_same_source(self):
        """The whole point: a re-OCR changes ``id`` (sha1 of the text) but not identity.

        Keying on the text digest would make a re-OCR read as an entirely new corpus,
        which Resume would then embed a second time.
        """
        before = {"id": hashlib.sha1(b"page one").hexdigest(), "metadata": {"Source-File": "/a/b.pdf"}}
        after = {"id": hashlib.sha1(b"page one, re-OCR'd").hexdigest(), "metadata": {"Source-File": "/a/b.pdf"}}
        self.assertNotEqual(before["id"], after["id"])
        self.assertEqual(
            names.source_digest(before["metadata"]["Source-File"]),
            names.source_digest(after["metadata"]["Source-File"]),
        )

    def test_differs_for_paths_that_derive_the_same_name(self):
        # The digest is provenance, so it must still tell apart what the name cannot.
        self.assertEqual(names.document_name("/a/case.pdf"), names.document_name("a/case.pdf"))
        self.assertNotEqual(names.source_digest("/a/case.pdf"), names.source_digest("a/case.pdf"))


class DigestFallbackTest(unittest.TestCase):
    """Obligation 16 -- the digest is the whole name when no usable path exists."""

    def test_empty_source_file(self):
        self.assertEqual(names.document_name(""), names.source_digest(""))

    def test_missing_source_file_is_the_same_case_as_empty(self):
        # A Record with no Source-File key arrives as None from a plain .get(); it must
        # derive the empty string's digest, so two such Records collide instead of one
        # overwriting the other.
        self.assertEqual(names.document_name(None), names.source_digest(""))

    def test_path_sanitizing_to_nothing(self):
        for source in ("/", "//", "..", ".", "/../", "./"):
            with self.subTest(source=source):
                self.assertEqual(names.document_name(source), names.source_digest(source))

    def test_over_long_component(self):
        source = "law/" + "x" * 256 + ".pdf"
        self.assertEqual(names.document_name(source), names.source_digest(source))

    def test_directory_component_at_the_cap_is_kept(self):
        # A directory gets no suffix appended, so it keeps the full 255 bytes.
        source = "law/" + "x" * 255 + "/doc.pdf"
        self.assertEqual(names.document_name(source), source)

    def test_filename_component_reserves_room_for_the_sink_suffix(self):
        # The Sink writes "<name>.npz" and "<name>.json", so the *filename* component
        # only gets 255 - len(".json"). At the cap it is kept; one byte over it falls
        # back to the digest rather than failing with ENAMETOOLONG at write time.
        cap = names.MAX_COMPONENT_BYTES - names.MAX_SUFFIX_BYTES
        at_cap = "law/" + "x" * cap
        self.assertEqual(names.document_name(at_cap), at_cap)
        over = "law/" + "x" * (cap + 1)
        self.assertEqual(names.document_name(over), names.source_digest(over))

    def test_a_derived_name_can_always_carry_both_sink_suffixes(self):
        # The regression this reserve exists for: a 255-byte source component used to be
        # kept verbatim, and "<name>.npz" then raised OSError(ENAMETOOLONG).
        with tempfile.TemporaryDirectory() as d:
            for source in ("law/" + "x" * 255, "law/" + "x" * 251 + ".pdf", "\u00e9" * 200 + ".pdf"):
                with self.subTest(source=source[:40]):
                    name = names.document_name(source)
                    base = Path(d) / name
                    base.parent.mkdir(parents=True, exist_ok=True)
                    for suffix in (".npz", ".json"):
                        base.with_name(base.name + suffix).write_bytes(b"x")

    def test_cap_is_measured_in_bytes_not_characters(self):
        # 200 characters, 400 bytes -- ext4 counts the encoded form, so this must fall back.
        source = "law/" + ("\u00e9" * 200) + ".pdf"
        self.assertLess(len(source), names.MAX_COMPONENT_BYTES)
        self.assertEqual(names.document_name(source), names.source_digest(source))

    def test_fallback_name_is_usable_as_a_filename(self):
        name = names.document_name("")
        self.assertNotIn("/", name)
        self.assertEqual(name + ".npz", f"{names.source_digest('')}.npz")


class CollisionTest(unittest.TestCase):
    """Obligation 17 -- the residue is fatal at startup, listing every group."""

    def test_leading_slash_collision_is_fatal(self):
        pairs = [
            (names.document_name("/a/case.pdf"), "/a/case.pdf"),
            (names.document_name("a/case.pdf"), "a/case.pdf"),
        ]
        with self.assertRaises(names.NameCollisionError) as ctx:
            names.check_collisions("law", pairs)
        message = str(ctx.exception)
        self.assertIn("'/a/case.pdf'", message)
        self.assertIn("'a/case.pdf'", message)
        self.assertIn("law", message)

    def test_tarball_collapse_is_fatal(self):
        gz, tar = "x.tar.gz::doc.pdf", "x.tar::doc.pdf"
        self.assertEqual(names.document_name(gz), names.document_name(tar))
        with self.assertRaises(names.NameCollisionError) as ctx:
            names.check_collisions("law", [(names.document_name(gz), gz), (names.document_name(tar), tar)])
        self.assertIn(gz, str(ctx.exception))
        self.assertIn(tar, str(ctx.exception))

    def test_two_empty_source_files_collide_on_the_digest(self):
        pairs = [(names.document_name(""), ""), (names.document_name(""), "")]
        with self.assertRaises(names.NameCollisionError):
            names.check_collisions("law", pairs)

    def test_every_colliding_group_is_listed(self):
        pairs = [
            (names.document_name("/a/case.pdf"), "/a/case.pdf"),
            (names.document_name("a/case.pdf"), "a/case.pdf"),
            (names.document_name("/b/brief.pdf"), "/b/brief.pdf"),
            (names.document_name("b/brief.pdf"), "b/brief.pdf"),
            (names.document_name("c/fine.pdf"), "c/fine.pdf"),
        ]
        with self.assertRaises(names.NameCollisionError) as ctx:
            names.check_collisions("law", pairs)
        message = str(ctx.exception)
        self.assertIn("a/case.pdf", message)
        self.assertIn("b/brief.pdf", message)
        self.assertNotIn("c/fine.pdf", message)

    def test_distinct_names_pass(self):
        pairs = [(names.document_name(s), s) for s in ("/a/case.pdf", "/a/brief.pdf", "b.pdf")]
        self.assertIsNone(names.check_collisions("law", pairs))

    def test_prevented_classes_never_reach_the_check(self):
        # Rule 3 (append) and rule 2 (normalize) already keep these apart.
        sources = ["law/case.pdf", "law/case.tiff", "/a/b/../c.pdf", "/a/b/c.pdf"]
        self.assertIsNone(names.check_collisions("law", [(names.document_name(s), s) for s in sources]))

    def test_the_same_name_in_two_runs_is_not_a_collision(self):
        # Scoped to one Run: both Sinks key on (run_label, document_name).
        pairs = [(names.document_name("a/case.pdf"), "a/case.pdf")]
        self.assertIsNone(names.check_collisions("qwen", pairs))
        self.assertIsNone(names.check_collisions("nemotron", pairs))


class ReservedNameTest(unittest.TestCase):
    """Obligation 22 -- landing on the reserved prefix is a collision, not an overwrite."""

    def test_reserved_root_name_is_fatal_even_when_unique(self):
        source = "/paperscale-embed"
        with self.assertRaises(names.NameCollisionError) as ctx:
            names.check_collisions("law", [(names.document_name(source), source)])
        self.assertIn(source, str(ctx.exception))

    def test_reserved_prefix_covers_the_manifest_and_the_failures_file(self):
        for source in ("/paperscale-embed.json", "paperscale-embed-failures.txt", "paperscale-embed.pdf"):
            with self.subTest(source=source):
                with self.assertRaises(names.NameCollisionError):
                    names.check_collisions("law", [(names.document_name(source), source)])

    def test_a_nested_reserved_name_is_allowed(self):
        # Only <out>/ holds the manifest, so "a/paperscale-embed.pdf.json" collides with
        # nothing. Rejecting it too would refuse Documents for no reason.
        source = "a/paperscale-embed.pdf"
        self.assertIsNone(names.check_collisions("law", [(names.document_name(source), source)]))
