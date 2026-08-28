"""Tests for the embed Record reader.

The reader exists only because `load_run` drops zero-length spans, so the
contrast against `load_run` is asserted here rather than described.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from paperscale.embed.records import DuplicateSourceFileError, iter_records, resolve_jsonl_paths
from paperscale.evaluation.runs import load_run
from tests.evaluation.fixtures import make_dolma_record, write_run


def _write_shard(directory: Path, name: str, records: list[dict]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


class ResolveJsonlPathsTest(unittest.TestCase):
    def test_workspace_dir_globs_results(self):
        with tempfile.TemporaryDirectory() as d:
            ws = write_run(Path(d), [make_dolma_record("/docs/a.pdf", ["x"])])
            self.assertEqual(resolve_jsonl_paths(ws), [Path(d) / "results" / "output_0.jsonl"])

    def test_bare_dir_of_jsonl(self):
        with tempfile.TemporaryDirectory() as d:
            _write_shard(Path(d), "run.jsonl", [make_dolma_record("/docs/a.pdf", ["x"])])
            self.assertEqual(resolve_jsonl_paths(d), [Path(d) / "run.jsonl"])

    def test_single_file(self):
        with tempfile.TemporaryDirectory() as d:
            f = write_run(Path(d), [make_dolma_record("/docs/a.pdf", ["x"])], as_workspace=False)
            self.assertEqual(resolve_jsonl_paths(f), [f])

    def test_str_and_path_agree(self):
        with tempfile.TemporaryDirectory() as d:
            ws = write_run(Path(d), [make_dolma_record("/docs/a.pdf", ["x"])])
            self.assertEqual(resolve_jsonl_paths(str(ws)), resolve_jsonl_paths(Path(ws)))

    def test_results_wins_over_top_level_jsonl(self):
        # A workspace can carry stray top-level .jsonl (a manifest, a hand-copied shard);
        # resolving to those instead of results/ would embed the wrong corpus silently.
        with tempfile.TemporaryDirectory() as d:
            ws = write_run(Path(d), [make_dolma_record("/docs/a.pdf", ["x"])])
            _write_shard(Path(d), "stray.jsonl", [make_dolma_record("/docs/b.pdf", ["y"])])
            self.assertEqual(resolve_jsonl_paths(ws), [Path(d) / "results" / "output_0.jsonl"])

    def test_shards_come_back_sorted(self):
        with tempfile.TemporaryDirectory() as d:
            results = Path(d) / "results"
            _write_shard(results, "output_1.jsonl", [make_dolma_record("/docs/b.pdf", ["y"])])
            _write_shard(results, "output_0.jsonl", [make_dolma_record("/docs/a.pdf", ["x"])])
            self.assertEqual([p.name for p in resolve_jsonl_paths(d)], ["output_0.jsonl", "output_1.jsonl"])

    def test_missing_path_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                resolve_jsonl_paths(Path(d) / "nope")

    def test_empty_dir_is_empty_not_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(resolve_jsonl_paths(d), [])


class IterRecordsTest(unittest.TestCase):
    def test_yields_whole_records_for_every_input_shape(self):
        rec = make_dolma_record("/docs/a.pdf", ["alpha", "beta"])
        with tempfile.TemporaryDirectory() as d:
            ws = write_run(Path(d), [rec])
            from_workspace = list(iter_records(ws))
        with tempfile.TemporaryDirectory() as d:
            _write_shard(Path(d), "run.jsonl", [rec])
            from_bare_dir = list(iter_records(d))
        with tempfile.TemporaryDirectory() as d:
            f = write_run(Path(d), [rec], as_workspace=False)
            from_file = list(iter_records(f))
        self.assertEqual(from_workspace, [rec])
        self.assertEqual(from_bare_dir, [rec])
        self.assertEqual(from_file, [rec])

    def test_zero_width_page_span_survives(self):
        # Page 2 is blank (natural_text None -> zero-width span). The packer needs it:
        # it costs 0 tokens, can never force a break, and still belongs to a Chunk's page range.
        with tempfile.TemporaryDirectory() as d:
            ws = write_run(Path(d), [make_dolma_record("/docs/a.pdf", ["alpha", "", "gamma"])])
            spans = list(iter_records(ws))[0]["attributes"]["pdf_page_numbers"]
        self.assertEqual([page for _, _, page in spans], [1, 2, 3])
        start, end, page = spans[1]
        self.assertEqual(page, 2)
        self.assertEqual(start, end)

    def test_load_run_drops_the_span_this_reader_keeps(self):
        # The premise of the whole module. If load_run ever stops dropping blank pages,
        # this fails first and points at the premise rather than at records.py.
        with tempfile.TemporaryDirectory() as d:
            ws = write_run(Path(d), [make_dolma_record("/docs/a.pdf", ["alpha", "", "gamma"])])
            pages, _ = load_run("m", ws)
            spans = list(iter_records(ws))[0]["attributes"]["pdf_page_numbers"]
        self.assertEqual([p.page for p in pages], [1, 3])
        self.assertEqual(len(spans), 3)

    def test_file_order_across_shards(self):
        with tempfile.TemporaryDirectory() as d:
            results = Path(d) / "results"
            _write_shard(results, "output_0.jsonl", [make_dolma_record("/docs/a.pdf", ["x"]), make_dolma_record("/docs/b.pdf", ["y"])])
            _write_shard(results, "output_1.jsonl", [make_dolma_record("/docs/c.pdf", ["z"])])
            names = [r["metadata"]["Source-File"] for r in iter_records(d)]
        self.assertEqual(names, ["/docs/a.pdf", "/docs/b.pdf", "/docs/c.pdf"])

    def test_blank_lines_are_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            rec = make_dolma_record("/docs/a.pdf", ["x"])
            path = Path(d) / "run.jsonl"
            path.write_text("\n" + json.dumps(rec) + "\n\n", encoding="utf-8")
            self.assertEqual(list(iter_records(path)), [rec])

    def test_duplicate_source_file_raises(self):
        with tempfile.TemporaryDirectory() as d:
            recs = [make_dolma_record("/docs/a.pdf", ["x"]), make_dolma_record("/docs/a.pdf", ["y"])]
            ws = write_run(Path(d), recs)
            with self.assertRaises(DuplicateSourceFileError) as caught:
                list(iter_records(ws))
        self.assertIn("/docs/a.pdf", str(caught.exception))

    def test_duplicate_across_shards_names_both_files(self):
        with tempfile.TemporaryDirectory() as d:
            results = Path(d) / "results"
            _write_shard(results, "output_0.jsonl", [make_dolma_record("/docs/a.pdf", ["x"])])
            _write_shard(results, "output_1.jsonl", [make_dolma_record("/docs/a.pdf", ["y"])])
            with self.assertRaises(DuplicateSourceFileError) as caught:
                list(iter_records(d))
        message = str(caught.exception)
        self.assertIn("output_0.jsonl", message)
        self.assertIn("output_1.jsonl", message)

    def test_missing_source_file_is_left_to_the_name_check(self):
        # names.py falls back to the digest for these; two of them must not be
        # reported here as a duplicate, or the same fault gets two wordings.
        with tempfile.TemporaryDirectory() as d:
            recs = [make_dolma_record("/docs/a.pdf", ["x"]), make_dolma_record("/docs/b.pdf", ["y"])]
            del recs[0]["metadata"]["Source-File"]
            recs[1]["metadata"]["Source-File"] = ""
            ws = write_run(Path(d), recs)
            self.assertEqual(len(list(iter_records(ws))), 2)

    def test_duplicate_is_scoped_to_one_run(self):
        # Two Runs may hold the same PDF; that is what the (run_label, document_name)
        # Resume key is for. Only a repeat inside one call is ambiguous.
        with tempfile.TemporaryDirectory() as d:
            a = write_run(Path(d) / "a", [make_dolma_record("/docs/a.pdf", ["x"])])
            b = write_run(Path(d) / "b", [make_dolma_record("/docs/a.pdf", ["y"])])
            self.assertEqual(len(list(iter_records(a))), 1)
            self.assertEqual(len(list(iter_records(b))), 1)


if __name__ == "__main__":
    unittest.main()
