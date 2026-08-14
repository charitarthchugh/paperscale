"""Tests for the SQLite eval store + leaderboard."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from paperscale.evaluation.db import EvalDB


def _line_for(text: str, model: str) -> str:
    for line in text.splitlines():
        if line.split()[:1] == [model]:
            return line
    raise AssertionError(f"no row for model {model!r} in:\n{text}")


class RoundTripTest(unittest.TestCase):
    def test_persists_across_reopen(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "eval.db"
            db = EvalDB(path)
            db.register_run("a", "/runs/a")
            db.write_correction_rate([("a", "doc1", 1, 0.1, 0.0), ("a", "doc1", 2, 0.3, 0.2)])
            db.close()

            db2 = EvalDB(path)
            rows = db2.conn.execute(
                "SELECT model, doc, page, correction_rate, uncorrectable_rate FROM correction_rate ORDER BY page"
            ).fetchall()
            db2.close()
        self.assertEqual(rows, [("a", "doc1", 1, 0.1, 0.0), ("a", "doc1", 2, 0.3, 0.2)])


class IdempotencyTest(unittest.TestCase):
    def test_rewriting_replaces_not_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            db = EvalDB(Path(d) / "eval.db")
            rows = [("a", "doc1", 1, 0.5, 0.1), ("a", "doc1", 2, 0.6, 0.2)]
            db.write_correction_rate(rows)
            db.write_correction_rate(rows)  # re-run
            n = db.conn.execute("SELECT COUNT(*) FROM correction_rate").fetchone()[0]
            db.close()
        self.assertEqual(n, 2)

    def test_rewriting_one_model_leaves_others(self):
        with tempfile.TemporaryDirectory() as d:
            db = EvalDB(Path(d) / "eval.db")
            db.write_correction_rate([("a", "doc1", 1, 0.5, 0.0)])
            db.write_correction_rate([("b", "doc1", 1, 0.8, 0.0)])
            db.write_correction_rate([("a", "doc1", 1, 0.9, 0.0)])  # rewrite a only
            rows = db.conn.execute(
                "SELECT model, correction_rate FROM correction_rate ORDER BY model"
            ).fetchall()
            db.close()
        self.assertEqual(rows, [("a", 0.9), ("b", 0.8)])


class LeaderboardTest(unittest.TestCase):
    def test_means_and_win_rate(self):
        # correction_rate is LOWER-better.
        # a: d1=.1 d2=.2 d3=.5  (mean 0.8/3 = 0.267); wins d1,d2 -> 2/3 = 0.67
        # b: d1=.3 d2=.4 d3=.1  (mean 0.8/3 = 0.267); wins d3     -> 1/3 = 0.33
        with tempfile.TemporaryDirectory() as d:
            db = EvalDB(Path(d) / "eval.db")
            db.write_correction_rate([
                ("a", "d1", 1, 0.1, 0.0), ("a", "d2", 1, 0.2, 0.0), ("a", "d3", 1, 0.5, 0.0),
                ("b", "d1", 1, 0.3, 0.0), ("b", "d2", 1, 0.4, 0.0), ("b", "d3", 1, 0.1, 0.0),
            ])
            out = db.leaderboard()
            db.close()
        self.assertIn("0.267", _line_for(out, "a"))  # hand-computed mean
        self.assertIn("0.67", _line_for(out, "a"))    # winner win-rate (lower better)
        self.assertIn("0.33", _line_for(out, "b"))

    def test_win_rate_restricted_to_common_docs(self):
        # a has an extra d4 that b lacks -> d4 must NOT count toward win-rate.
        # Common docs = {d1,d2,d3}; a wins d1,d2 (lower) -> 2/3 = 0.67 (NOT 3/4 = 0.75).
        with tempfile.TemporaryDirectory() as d:
            db = EvalDB(Path(d) / "eval.db")
            db.write_correction_rate([
                ("a", "d1", 1, 0.1, 0.0), ("a", "d2", 1, 0.2, 0.0), ("a", "d3", 1, 0.5, 0.0),
                ("a", "d4", 1, 0.01, 0.0),  # b lacks d4
                ("b", "d1", 1, 0.3, 0.0), ("b", "d2", 1, 0.4, 0.0), ("b", "d3", 1, 0.1, 0.0),
            ])
            out = db.leaderboard()
            db.close()
        a_line = _line_for(out, "a")
        self.assertIn("0.67", a_line)
        self.assertNotIn("0.75", a_line)

    def test_leaderboard_mean_is_doc_weighted_not_page_weighted(self):
        # d1 has 3 pages @0.9, d2 has 1 page @0.0. Doc-mean = (0.9+0.0)/2 = 0.450,
        # NOT the page-weighted 0.675 = (0.9*3 + 0.0)/4 -- a big doc must not dominate.
        with tempfile.TemporaryDirectory() as d:
            db = EvalDB(Path(d) / "eval.db")
            db.write_correction_rate([
                ("a", "d1", 1, 0.9, 0.0), ("a", "d1", 2, 0.9, 0.0), ("a", "d1", 3, 0.9, 0.0),
                ("a", "d2", 1, 0.0, 0.0),
            ])
            out = db.leaderboard()
            db.close()
        line = _line_for(out, "a")
        self.assertIn("0.450", line)
        self.assertNotIn("0.675", line)

    def test_single_model_win_rate_na(self):
        with tempfile.TemporaryDirectory() as d:
            db = EvalDB(Path(d) / "eval.db")
            db.write_correction_rate([("a", "d1", 1, 0.5, 0.0)])
            out = db.leaderboard()
            db.close()
        self.assertIn("n/a", _line_for(out, "a"))


class PplxTest(unittest.TestCase):
    def test_pplx_table_created_and_shown(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "eval.db"
            db = EvalDB(path)
            db.write_correction_rate([("my-model", "d1", 1, 0.5, 0.0)])
            # rows: (doc, page, n_tok_raw, sum_lp_raw, ppl_raw, n_tok_c, sum_lp_c, ppl_c)
            db.write_pplx_doc("my-model", [
                ("d1", 1, 10, -5.0, 4.0, 10, -4.0, 2.0),
                ("d1", 2, 10, -5.0, 6.0, 10, -4.0, 4.0),
            ])
            tables = [r[0] for r in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'pplx_%'"
            ).fetchall()]
            out = db.leaderboard()
            db.close()
        self.assertIn("pplx_my_model", tables)
        self.assertIn("ppl_raw", out)     # column header present
        self.assertIn("5.000", out)       # mean ppl_raw = (4+6)/2 = 5
        self.assertIn("3.000", out)       # mean ppl_corrected = (2+4)/2 = 3

    def test_pplx_rewrite_replaces(self):
        with tempfile.TemporaryDirectory() as d:
            db = EvalDB(Path(d) / "eval.db")
            db.write_pplx_doc("m", [("d1", 1, 1, -1.0, 2.0, 1, -1.0, 2.0)])
            db.write_pplx_doc("m", [("d1", 1, 1, -1.0, 9.0, 1, -1.0, 9.0)])
            n = db.conn.execute("SELECT COUNT(*) FROM pplx_m").fetchone()[0]
            db.close()
        self.assertEqual(n, 1)

    def test_resume_tracks_done_docs_and_clear(self):
        with tempfile.TemporaryDirectory() as d:
            db = EvalDB(Path(d) / "eval.db")
            self.assertEqual(db.pplx_done_docs("m"), set())
            db.write_pplx_doc("m", [("d1", 1, 1, -1.0, 2.0, 1, -1.0, 2.0)])
            db.write_pplx_doc("m", [("d2", 1, 1, -1.0, 2.0, 1, -1.0, 2.0)])
            self.assertEqual(db.pplx_done_docs("m"), {"d1", "d2"})
            db.clear_pplx("m")
            self.assertEqual(db.pplx_done_docs("m"), set())
            db.close()

    def test_scorer_model_change_drops_pplx_table(self):
        with tempfile.TemporaryDirectory() as d:
            db = EvalDB(Path(d) / "eval.db")
            db.register_run("m", "/in", "scorer-a")
            db.write_pplx_doc("m", [("d1", 1, 1, -1.0, 2.0, 1, -1.0, 2.0)])
            # No --pplx this run: stored scorer (and scores) survive.
            db.register_run("m", "/in", None)
            self.assertEqual(db.pplx_done_docs("m"), {"d1"})
            # Same scorer: scores survive (resume).
            db.register_run("m", "/in", "scorer-a")
            self.assertEqual(db.pplx_done_docs("m"), {"d1"})
            # Different scorer: stale scores dropped.
            db.register_run("m", "/in", "scorer-b")
            self.assertEqual(db.pplx_done_docs("m"), set())
            db.close()


class ResumeBookkeepingTest(unittest.TestCase):
    """eval_doc records which (model, doc, phase) triples are complete, keyed by a
    checksum of the doc's text so a re-OCR'd run invalidates itself per document."""

    def test_doc_is_done_when_checksum_matches(self):
        with tempfile.TemporaryDirectory() as d:
            db = EvalDB(Path(d) / "eval.db")
            db.mark_doc_done("a", "d1", "corrections", "sha-1")
            done = db.done_docs("a", "corrections", {"d1": "sha-1"})
            db.close()
        self.assertEqual(done, {"d1"})

    def test_doc_is_not_done_when_checksum_differs(self):
        # The run was re-OCR'd under the same label -> the cached rows are stale.
        with tempfile.TemporaryDirectory() as d:
            db = EvalDB(Path(d) / "eval.db")
            db.mark_doc_done("a", "d1", "corrections", "sha-1")
            done = db.done_docs("a", "corrections", {"d1": "sha-2"})
            db.close()
        self.assertEqual(done, set())

    def test_unmarked_doc_is_not_done(self):
        with tempfile.TemporaryDirectory() as d:
            db = EvalDB(Path(d) / "eval.db")
            done = db.done_docs("a", "corrections", {"d1": "sha-1"})
            db.close()
        self.assertEqual(done, set())

    def test_phases_are_tracked_independently(self):
        # A crash between phases leaves corrections done but textlayer not.
        with tempfile.TemporaryDirectory() as d:
            db = EvalDB(Path(d) / "eval.db")
            db.mark_doc_done("a", "d1", "corrections", "sha-1")
            corrections = db.done_docs("a", "corrections", {"d1": "sha-1"})
            textlayer = db.done_docs("a", "textlayer", {"d1": "sha-1"})
            db.close()
        self.assertEqual(corrections, {"d1"})
        self.assertEqual(textlayer, set())

    def test_models_are_tracked_independently(self):
        with tempfile.TemporaryDirectory() as d:
            db = EvalDB(Path(d) / "eval.db")
            db.mark_doc_done("a", "d1", "corrections", "sha-1")
            done = db.done_docs("b", "corrections", {"d1": "sha-1"})
            db.close()
        self.assertEqual(done, set())

    def test_doc_that_produced_no_rows_is_still_done(self):
        # textlayer writes zero rows for a blank-layer doc, but the pdftotext calls
        # were already paid for -- row presence must not be what "done" means.
        with tempfile.TemporaryDirectory() as d:
            db = EvalDB(Path(d) / "eval.db")
            db.mark_doc_done("a", "blank-doc", "textlayer", "sha-1")
            done = db.done_docs("a", "textlayer", {"blank-doc": "sha-1"})
            n = db.conn.execute("SELECT COUNT(*) FROM textlayer_agreement").fetchone()[0]
            db.close()
        self.assertEqual(done, {"blank-doc"})
        self.assertEqual(n, 0)

    def test_remarking_a_doc_replaces_its_checksum(self):
        with tempfile.TemporaryDirectory() as d:
            db = EvalDB(Path(d) / "eval.db")
            db.mark_doc_done("a", "d1", "corrections", "sha-1")
            db.mark_doc_done("a", "d1", "corrections", "sha-2")
            n = db.conn.execute("SELECT COUNT(*) FROM eval_doc").fetchone()[0]
            done = db.done_docs("a", "corrections", {"d1": "sha-2"})
            db.close()
        self.assertEqual(n, 1)
        self.assertEqual(done, {"d1"})


class PerDocWriteTest(unittest.TestCase):
    """write_doc replaces one doc's rows and marks it done in a single transaction,
    so an interrupted phase keeps every doc that finished."""

    def test_writes_one_doc_without_touching_siblings(self):
        with tempfile.TemporaryDirectory() as d:
            db = EvalDB(Path(d) / "eval.db")
            db.write_doc("corrections", "a", "d1", [("a", "d1", 1, 0.1, 0.0)], "sha-1")
            db.write_doc("corrections", "a", "d2", [("a", "d2", 1, 0.2, 0.0)], "sha-2")
            rows = db.conn.execute(
                "SELECT doc, correction_rate FROM correction_rate ORDER BY doc"
            ).fetchall()
            db.close()
        self.assertEqual(rows, [("d1", 0.1), ("d2", 0.2)])

    def test_rewriting_a_doc_replaces_only_that_doc(self):
        with tempfile.TemporaryDirectory() as d:
            db = EvalDB(Path(d) / "eval.db")
            db.write_doc("corrections", "a", "d1", [("a", "d1", 1, 0.1, 0.0)], "sha-1")
            db.write_doc("corrections", "a", "d2", [("a", "d2", 1, 0.2, 0.0)], "sha-2")
            db.write_doc("corrections", "a", "d1", [("a", "d1", 1, 0.9, 0.0)], "sha-9")
            rows = db.conn.execute(
                "SELECT doc, correction_rate FROM correction_rate ORDER BY doc"
            ).fetchall()
            db.close()
        self.assertEqual(rows, [("d1", 0.9), ("d2", 0.2)])

    def test_write_marks_the_doc_done(self):
        with tempfile.TemporaryDirectory() as d:
            db = EvalDB(Path(d) / "eval.db")
            db.write_doc("corrections", "a", "d1", [("a", "d1", 1, 0.1, 0.0)], "sha-1")
            done = db.done_docs("a", "corrections", {"d1": "sha-1"})
            db.close()
        self.assertEqual(done, {"d1"})

    def test_each_doc_is_committed_so_an_interrupted_phase_survives(self):
        # Simulates a crash: write two docs, never call close(), reopen the file.
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "eval.db"
            db = EvalDB(path)
            db.write_doc("corrections", "a", "d1", [("a", "d1", 1, 0.1, 0.0)], "sha-1")
            db.write_doc("corrections", "a", "d2", [("a", "d2", 1, 0.2, 0.0)], "sha-2")
            del db  # no close(), no explicit commit

            db2 = EvalDB(path)
            done = db2.done_docs("a", "corrections", {"d1": "sha-1", "d2": "sha-2", "d3": "sha-3"})
            db2.close()
        self.assertEqual(done, {"d1", "d2"})

    def test_failed_write_leaves_the_doc_not_done(self):
        # Atomicity: a doc must never be marked done with no rows behind it, or
        # resume would skip work that never happened.
        with tempfile.TemporaryDirectory() as d:
            db = EvalDB(Path(d) / "eval.db")
            with self.assertRaises(sqlite3.ProgrammingError):
                db.write_doc("corrections", "a", "d1", [("wrong", "arity")], "sha-1")
            done = db.done_docs("a", "corrections", {"d1": "sha-1"})
            n = db.conn.execute("SELECT COUNT(*) FROM correction_rate").fetchone()[0]
            db.close()
        self.assertEqual(done, set())
        self.assertEqual(n, 0)

    def test_zero_rows_still_marks_done(self):
        # textlayer's blank-layer docs: no rows, but the work is complete.
        with tempfile.TemporaryDirectory() as d:
            db = EvalDB(Path(d) / "eval.db")
            db.write_doc("textlayer", "a", "blank", [], "sha-1")
            done = db.done_docs("a", "textlayer", {"blank": "sha-1"})
            db.close()
        self.assertEqual(done, {"blank"})

    def test_unknown_phase_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            db = EvalDB(Path(d) / "eval.db")
            with self.assertRaises(KeyError):
                db.write_doc("nonsense", "a", "d1", [], "sha-1")
            db.close()


class ClearModelTest(unittest.TestCase):
    """--no-resume: drop everything cached for a label so it rescores from scratch."""

    def _populate(self, db):
        db.write_doc("corrections", "a", "d1", [("a", "d1", 1, 0.1, 0.0)], "sha-a")
        db.write_doc("corrections", "b", "d1", [("b", "d1", 1, 0.2, 0.0)], "sha-b")
        db.write_doc("garbage", "a", "d1", [("a", "d1", 1, 0.3)], "sha-a")
        db.write_doc("textlayer", "a", "d1", [("a", "d1", 1, 0.4, 0.5)], "sha-a")
        db.write_reject_rate([("a", "d1", 0, 3), ("b", "d1", 1, 3)])
        db.write_pplx_doc("a", [("d1", 1, 1, -1.0, 2.0, 1, -1.0, 2.0)])
        db.write_peer_agreement([
            ("a", "b", "d1", 1, 0.7, 0.8),
            ("b", "a", "d1", 1, 0.7, 0.8),
        ])

    def test_clears_every_metric_for_that_model(self):
        with tempfile.TemporaryDirectory() as d:
            db = EvalDB(Path(d) / "eval.db")
            self._populate(db)
            db.clear_model("a")
            counts = {
                t: db.conn.execute(f"SELECT COUNT(*) FROM {t} WHERE model = 'a'").fetchone()[0]
                for t in ("correction_rate", "garbage_fraction", "textlayer_agreement",
                          "reject_rate", "eval_doc")
            }
            db.close()
        self.assertEqual(counts, dict.fromkeys(counts, 0))

    def test_clears_peer_rows_in_both_directions(self):
        # (b, a) is b's score AGAINST a. If a is rescored that row is stale too.
        with tempfile.TemporaryDirectory() as d:
            db = EvalDB(Path(d) / "eval.db")
            self._populate(db)
            db.clear_model("a")
            rows = db.conn.execute("SELECT model, peer FROM peer_agreement").fetchall()
            db.close()
        self.assertEqual(rows, [])

    def test_clears_pplx_scores(self):
        with tempfile.TemporaryDirectory() as d:
            db = EvalDB(Path(d) / "eval.db")
            self._populate(db)
            db.clear_model("a")
            done = db.pplx_done_docs("a")
            db.close()
        self.assertEqual(done, set())

    def test_leaves_other_models_intact(self):
        with tempfile.TemporaryDirectory() as d:
            db = EvalDB(Path(d) / "eval.db")
            self._populate(db)
            db.clear_model("a")
            corr = db.conn.execute("SELECT doc, correction_rate FROM correction_rate WHERE model='b'").fetchall()
            done = db.done_docs("b", "corrections", {"d1": "sha-b"})
            db.close()
        self.assertEqual(corr, [("d1", 0.2)])
        self.assertEqual(done, {"d1"})


class PruneMissingDocsTest(unittest.TestCase):
    """A doc in the DB but absent from the run (you deleted the PDF) must go, or
    the leaderboard keeps averaging it in."""

    def test_drops_docs_absent_from_the_run(self):
        with tempfile.TemporaryDirectory() as d:
            db = EvalDB(Path(d) / "eval.db")
            db.write_doc("corrections", "a", "keep", [("a", "keep", 1, 0.1, 0.0)], "sha-1")
            db.write_doc("corrections", "a", "gone", [("a", "gone", 1, 0.2, 0.0)], "sha-2")
            db.prune_missing_docs("a", {"keep"})
            rows = db.conn.execute("SELECT doc FROM correction_rate WHERE model='a'").fetchall()
            done = db.done_docs("a", "corrections", {"keep": "sha-1", "gone": "sha-2"})
            db.close()
        self.assertEqual(rows, [("keep",)])
        self.assertEqual(done, {"keep"})

    def test_drops_peer_rows_for_the_missing_doc_in_both_directions(self):
        with tempfile.TemporaryDirectory() as d:
            db = EvalDB(Path(d) / "eval.db")
            db.write_peer_agreement([
                ("a", "b", "gone", 1, 0.7, 0.8),
                ("b", "a", "gone", 1, 0.7, 0.8),
                ("a", "b", "keep", 1, 0.9, 0.9),
            ])
            db.prune_missing_docs("a", {"keep"})
            rows = db.conn.execute("SELECT model, peer, doc FROM peer_agreement ORDER BY model").fetchall()
            db.close()
        self.assertEqual(rows, [("a", "b", "keep")])

    def test_leaves_other_models_docs_alone(self):
        with tempfile.TemporaryDirectory() as d:
            db = EvalDB(Path(d) / "eval.db")
            db.write_doc("corrections", "a", "gone", [("a", "gone", 1, 0.2, 0.0)], "sha-1")
            db.write_doc("corrections", "b", "gone", [("b", "gone", 1, 0.3, 0.0)], "sha-2")
            db.prune_missing_docs("a", {"keep"})
            rows = db.conn.execute("SELECT model, doc FROM correction_rate").fetchall()
            db.close()
        self.assertEqual(rows, [("b", "gone")])


class StoredPeerPairsTest(unittest.TestCase):
    """peer_agreement is its own resume bookkeeping: the rows on disk say which
    pairs are done. eval_doc's 'peer' entry only records the text checksum, so a
    changed doc can have its pairs invalidated."""

    def test_reports_pairs_keyed_by_page(self):
        with tempfile.TemporaryDirectory() as d:
            db = EvalDB(Path(d) / "eval.db")
            db.write_peer_agreement([
                ("a", "b", "d1", 1, 0.7, 0.8),
                ("b", "a", "d1", 1, 0.7, 0.8),
                ("a", "b", "d1", 2, 0.5, 0.5),
                ("b", "a", "d1", 2, 0.5, 0.5),
            ])
            pairs = db.stored_peer_pairs()
            db.close()
        self.assertEqual(pairs, {
            ("d1", 1): {("a", "b"), ("b", "a")},
            ("d1", 2): {("a", "b"), ("b", "a")},
        })

    def test_empty_db_reports_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            db = EvalDB(Path(d) / "eval.db")
            pairs = db.stored_peer_pairs()
            db.close()
        self.assertEqual(pairs, {})

    def test_clearing_a_changed_doc_drops_both_directions(self):
        # Model a's d1 was re-OCR'd: every pair touching a on d1 is stale, including
        # the rows where a is the peer rather than the model.
        with tempfile.TemporaryDirectory() as d:
            db = EvalDB(Path(d) / "eval.db")
            db.write_peer_agreement([
                ("a", "b", "d1", 1, 0.7, 0.8),
                ("b", "a", "d1", 1, 0.7, 0.8),
                ("b", "c", "d1", 1, 0.6, 0.6),
                ("c", "b", "d1", 1, 0.6, 0.6),
                ("a", "b", "d2", 1, 0.9, 0.9),
                ("b", "a", "d2", 1, 0.9, 0.9),
            ])
            db.clear_peer_docs([("a", "d1")])
            pairs = db.stored_peer_pairs()
            db.close()
        # b-c on d1 survives (does not involve a); d1's a-b pairs are gone; d2 untouched.
        self.assertEqual(pairs, {
            ("d1", 1): {("b", "c"), ("c", "b")},
            ("d2", 1): {("a", "b"), ("b", "a")},
        })


class BatchBookkeepingTest(unittest.TestCase):
    """The peer phase touches every doc on every run, so its bookkeeping commits
    once rather than once per doc (one fsync per doc would stall a warm no-op)."""

    def test_mark_docs_done_records_every_row(self):
        with tempfile.TemporaryDirectory() as d:
            db = EvalDB(Path(d) / "eval.db")
            db.mark_docs_done([("a", "d1", "peer", "s1"), ("a", "d2", "peer", "s2"),
                               ("b", "d1", "peer", "s3")])
            a = db.done_docs("a", "peer", {"d1": "s1", "d2": "s2"})
            b = db.done_docs("b", "peer", {"d1": "s3"})
            db.close()
        self.assertEqual(a, {"d1", "d2"})
        self.assertEqual(b, {"d1"})

    def test_mark_docs_done_replaces_existing_checksums(self):
        with tempfile.TemporaryDirectory() as d:
            db = EvalDB(Path(d) / "eval.db")
            db.mark_docs_done([("a", "d1", "peer", "old")])
            db.mark_docs_done([("a", "d1", "peer", "new")])
            n = db.conn.execute("SELECT COUNT(*) FROM eval_doc").fetchone()[0]
            done = db.done_docs("a", "peer", {"d1": "new"})
            db.close()
        self.assertEqual(n, 1)
        self.assertEqual(done, {"d1"})

    def test_clear_peer_docs_drops_both_directions_for_each_pair(self):
        with tempfile.TemporaryDirectory() as d:
            db = EvalDB(Path(d) / "eval.db")
            db.write_peer_agreement([
                ("a", "b", "d1", 1, 0.7, 0.8), ("b", "a", "d1", 1, 0.7, 0.8),
                ("a", "b", "d2", 1, 0.9, 0.9), ("b", "a", "d2", 1, 0.9, 0.9),
            ])
            db.clear_peer_docs([("a", "d1")])
            pairs = db.stored_peer_pairs()
            db.close()
        self.assertEqual(pairs, {("d2", 1): {("a", "b"), ("b", "a")}})

    def test_empty_batches_are_noops(self):
        with tempfile.TemporaryDirectory() as d:
            db = EvalDB(Path(d) / "eval.db")
            db.mark_docs_done([])
            db.clear_peer_docs([])
            n = db.conn.execute("SELECT COUNT(*) FROM eval_doc").fetchone()[0]
            db.close()
        self.assertEqual(n, 0)


class AppendPeerAgreementTest(unittest.TestCase):
    def test_append_keeps_pairs_written_by_an_earlier_run(self):
        # Resume computes only the MISSING pairs, so the writer must add rather than
        # replace -- _replace's delete-by-model would wipe the pairs being reused.
        with tempfile.TemporaryDirectory() as d:
            db = EvalDB(Path(d) / "eval.db")
            db.write_peer_agreement([("a", "b", "d1", 1, 0.7, 0.8), ("b", "a", "d1", 1, 0.7, 0.8)])
            db.append_peer_agreement([("a", "c", "d1", 1, 0.6, 0.6), ("c", "a", "d1", 1, 0.6, 0.6)])
            pairs = {(r[0], r[1]) for r in db.conn.execute("SELECT model, peer FROM peer_agreement")}
            db.close()
        self.assertEqual(pairs, {("a", "b"), ("b", "a"), ("a", "c"), ("c", "a")})

    def test_append_of_nothing_is_a_noop(self):
        with tempfile.TemporaryDirectory() as d:
            db = EvalDB(Path(d) / "eval.db")
            db.append_peer_agreement([])
            n = db.conn.execute("SELECT COUNT(*) FROM peer_agreement").fetchone()[0]
            db.close()
        self.assertEqual(n, 0)


class MarkedDocsTest(unittest.TestCase):
    def test_reports_docs_with_a_checksum_record_regardless_of_value(self):
        # Distinct from done_docs: this ignores whether the checksum still matches.
        # Used to spot pplx scores written before eval_doc existed.
        with tempfile.TemporaryDirectory() as d:
            db = EvalDB(Path(d) / "eval.db")
            db.mark_doc_done("a", "d1", "pplx", "old-sha")
            marked = db.marked_docs("a", "pplx")
            done = db.done_docs("a", "pplx", {"d1": "new-sha"})
            db.close()
        self.assertEqual(marked, {"d1"})
        self.assertEqual(done, set())


if __name__ == "__main__":
    unittest.main()
