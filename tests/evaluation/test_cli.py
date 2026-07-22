"""End-to-end test of `paperscale evaluate` (non-pplx path)."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from paperscale.cli import _parse_runs, main
from tests.evaluation.fixtures import make_dolma_record, write_run

CLEAN = ["The quick brown fox jumps over the lazy dog.", "Every good boy deserves fruit today."]
GARBLED = ["Teh qckuz brwn fxo jmps ovr teh lzy dg.", "Evry gd boy dsrvs frt3 tdy xkcdqz."]


class ParseRunsTest(unittest.TestCase):
    def test_rejects_missing_equals(self):
        with self.assertRaises(SystemExit):
            _parse_runs(["nopath"])

    def test_rejects_duplicate_label(self):
        with self.assertRaises(SystemExit):
            _parse_runs(["a=/x", "a=/y"])


class EvaluateE2ETest(unittest.TestCase):
    def test_two_runs_ranks_good_first(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            good = write_run(root / "good", [make_dolma_record("/docs/a.pdf", CLEAN)])
            bad = write_run(root / "bad", [make_dolma_record("/docs/a.pdf", GARBLED)])
            db_path = root / "eval.sqlite"

            rc = main(["evaluate", "--run", f"good={good}", "--run", f"bad={bad}", "--db", str(db_path)])
            self.assertEqual(rc, 0)

            con = sqlite3.connect(db_path)
            # both runs registered
            self.assertEqual({r[0] for r in con.execute("SELECT model FROM runs")}, {"good", "bad"})
            # good's text needs fewer spell-corrections than bad's (lower is better)
            means = dict(con.execute("SELECT model, AVG(correction_rate) FROM correction_rate GROUP BY model"))
            self.assertLess(means["good"], means["bad"])
            # peer agreement rows exist (2 models, shared doc/pages)
            self.assertGreater(con.execute("SELECT COUNT(*) FROM peer_agreement").fetchone()[0], 0)
            con.close()

    def test_single_run_skips_peer_agreement(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            good = write_run(root / "good", [make_dolma_record("/docs/a.pdf", CLEAN)])
            rc = main(["evaluate", "--run", f"good={good}", "--db", str(root / "eval.sqlite")])
            self.assertEqual(rc, 0)
            con = sqlite3.connect(root / "eval.sqlite")
            self.assertEqual(con.execute("SELECT COUNT(*) FROM peer_agreement").fetchone()[0], 0)
            con.close()


if __name__ == "__main__":
    unittest.main()
