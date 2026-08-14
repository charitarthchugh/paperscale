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


SENTINEL = 0.12345  # a value no real metric would produce


class ResumeE2ETest(unittest.TestCase):
    """Resume is observed by poisoning a cached row and re-running: if the phase
    was skipped the poisoned value survives, if it was recomputed it is replaced.
    This works across the process pool, which monkeypatching cannot."""

    def _evaluate(self, *runs: str, db: Path, extra: list[str] | None = None) -> int:
        argv = ["evaluate"]
        for r in runs:
            argv += ["--run", r]
        argv += ["--db", str(db)]
        return main(argv + (extra or []))

    def _poison(self, db: Path, model: str) -> None:
        con = sqlite3.connect(db)
        con.execute("UPDATE correction_rate SET correction_rate = ? WHERE model = ?", (SENTINEL, model))
        con.commit()
        con.close()

    def _corrections(self, db: Path, model: str) -> list[float]:
        con = sqlite3.connect(db)
        vals = [r[0] for r in con.execute("SELECT correction_rate FROM correction_rate WHERE model = ?", (model,))]
        con.close()
        return vals

    def test_unchanged_run_is_not_rescored(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run = write_run(root / "good", [make_dolma_record("/docs/a.pdf", CLEAN)])
            db = root / "eval.sqlite"
            self._evaluate(f"good={run}", db=db)
            self._poison(db, "good")
            self._evaluate(f"good={run}", db=db)
            vals = self._corrections(db, "good")
        self.assertTrue(vals)
        self.assertEqual(set(vals), {SENTINEL})  # untouched -> phase was skipped

    def test_changed_run_is_rescored(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            db = root / "eval.sqlite"
            write_run(root / "good", [make_dolma_record("/docs/a.pdf", CLEAN)])
            self._evaluate(f"good={root / 'good'}", db=db)
            self._poison(db, "good")
            # Re-OCR under the same label: the checksum changes, so the cache is stale.
            write_run(root / "good", [make_dolma_record("/docs/a.pdf", GARBLED)])
            self._evaluate(f"good={root / 'good'}", db=db)
            vals = self._corrections(db, "good")
        self.assertTrue(vals)
        self.assertNotIn(SENTINEL, vals)

    def test_no_resume_rescores_everything(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run = write_run(root / "good", [make_dolma_record("/docs/a.pdf", CLEAN)])
            db = root / "eval.sqlite"
            self._evaluate(f"good={run}", db=db)
            self._poison(db, "good")
            self._evaluate(f"good={run}", db=db, extra=["--no-resume"])
            vals = self._corrections(db, "good")
        self.assertTrue(vals)
        self.assertNotIn(SENTINEL, vals)

    def test_zero_row_phase_is_still_recorded_as_done(self):
        # /docs/a.pdf does not exist -> textlayer skips the doc and writes no rows,
        # but the phase must still count as complete.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run = write_run(root / "good", [make_dolma_record("/docs/a.pdf", CLEAN)])
            db = root / "eval.sqlite"
            self._evaluate(f"good={run}", db=db)
            con = sqlite3.connect(db)
            n_rows = con.execute("SELECT COUNT(*) FROM textlayer_agreement").fetchone()[0]
            phases = {r[0] for r in con.execute("SELECT phase FROM eval_doc WHERE model='good'")}
            con.close()
        self.assertEqual(n_rows, 0)
        self.assertIn("textlayer", phases)

    def test_adding_a_run_keeps_existing_peer_rows(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            db = root / "eval.sqlite"
            a = write_run(root / "a", [make_dolma_record("/docs/a.pdf", CLEAN)])
            b = write_run(root / "b", [make_dolma_record("/docs/a.pdf", GARBLED)])
            c = write_run(root / "c", [make_dolma_record("/docs/a.pdf", CLEAN)])
            self._evaluate(f"a={a}", f"b={b}", db=db)

            con = sqlite3.connect(db)
            con.execute("UPDATE peer_agreement SET bow_f1 = ? WHERE model='a' AND peer='b'", (SENTINEL,))
            con.commit()
            con.close()

            self._evaluate(f"a={a}", f"b={b}", f"c={c}", db=db)

            con = sqlite3.connect(db)
            ab = [r[0] for r in con.execute("SELECT bow_f1 FROM peer_agreement WHERE model='a' AND peer='b'")]
            pairs = {(r[0], r[1]) for r in con.execute("SELECT model, peer FROM peer_agreement")}
            con.close()
        self.assertEqual(set(ab), {SENTINEL})  # a-b reused, not rescored
        self.assertIn(("a", "c"), pairs)       # new pairs filled in
        self.assertIn(("b", "c"), pairs)

    def test_resumed_run_matches_a_fresh_run(self):
        # The property that makes resume trustworthy: reusing cached scores must give
        # byte-identical output to computing everything in one pass.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            docs = [make_dolma_record("/docs/a.pdf", CLEAN), make_dolma_record("/docs/b.pdf", GARBLED)]
            a = write_run(root / "a", docs)
            b = write_run(root / "b", [make_dolma_record("/docs/a.pdf", GARBLED),
                                      make_dolma_record("/docs/b.pdf", CLEAN)])

            # Fresh: one pass, everything computed.
            fresh_db = root / "fresh.sqlite"
            self._evaluate(f"a={a}", f"b={b}", db=fresh_db, extra=["--no-resume"])

            # Resumed: a alone, then both, then both again. The third call must be a
            # pure no-op -- if append_peer_agreement re-added pairs it would double them.
            resumed_db = root / "resumed.sqlite"
            self._evaluate(f"a={a}", db=resumed_db)
            self._evaluate(f"a={a}", f"b={b}", db=resumed_db)
            self._evaluate(f"a={a}", f"b={b}", db=resumed_db)

            rows = {}
            for name, path in (("fresh", fresh_db), ("resumed", resumed_db)):
                con = sqlite3.connect(path)
                rows[name] = {
                    t: sorted(con.execute(f"SELECT * FROM {t}"))
                    for t in ("correction_rate", "garbage_fraction", "peer_agreement",
                              "textlayer_agreement", "reject_rate")
                }
                con.close()
        for table in rows["fresh"]:
            self.assertEqual(rows["resumed"][table], rows["fresh"][table], f"{table} differs")

    def test_doc_removed_from_run_is_pruned(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            db = root / "eval.sqlite"
            write_run(root / "good", [
                make_dolma_record("/docs/a.pdf", CLEAN),
                make_dolma_record("/docs/b.pdf", CLEAN),
            ])
            self._evaluate(f"good={root / 'good'}", db=db)
            write_run(root / "good", [make_dolma_record("/docs/a.pdf", CLEAN)])
            self._evaluate(f"good={root / 'good'}", db=db)

            con = sqlite3.connect(db)
            docs = {r[0] for r in con.execute("SELECT DISTINCT doc FROM correction_rate")}
            stale = {r[0] for r in con.execute("SELECT DISTINCT doc FROM eval_doc")}
            con.close()
        self.assertEqual(docs, {"/docs/a.pdf"})
        self.assertEqual(stale, {"/docs/a.pdf"})


if __name__ == "__main__":
    unittest.main()
