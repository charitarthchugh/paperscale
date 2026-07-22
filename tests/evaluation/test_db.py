"""Tests for the SQLite eval store + leaderboard."""

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
            db.write_pplx("my-model", [
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
            db.write_pplx("m", [("d1", 1, 1, -1.0, 2.0, 1, -1.0, 2.0)])
            db.write_pplx("m", [("d1", 1, 1, -1.0, 9.0, 1, -1.0, 9.0)])
            n = db.conn.execute("SELECT COUNT(*) FROM pplx_m").fetchone()[0]
            db.close()
        self.assertEqual(n, 1)


if __name__ == "__main__":
    unittest.main()
