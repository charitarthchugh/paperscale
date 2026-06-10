from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from paperscale.candidates import CandidateRecord, plan_candidates, read_candidates

FIXED_CLOCK = lambda: 1700000000.0  # noqa: E731 - tiny deterministic clock for tests


def _touch(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class PlanCandidatesTests(unittest.TestCase):
    def _common(self, state_root: Path, output_dir: Path, inputs: list[Path]):
        return plan_candidates(
            inputs,
            output_dir=output_dir,
            state_root=state_root,
            profile="default",
            model=None,
            base_url="http://fake/v1",
            capacity="m",
            clock=FIXED_CLOCK,
        )

    def test_abspath_dedup_drops_exact_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_root = root / ".paperscale"
            out_dir = root / "out"
            doc = _touch(root / "doc.pdf")
            # Same file passed twice, once absolute and once via a relative-ish form.
            inputs = [doc, doc, Path(tmp) / "doc.pdf"]

            path = self._common(state_root, out_dir, inputs)
            records = read_candidates(path)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].input_path, str(doc.resolve()))

    def test_stem_dedup_across_different_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_root = root / ".paperscale"
            out_dir = root / "out"
            a = _touch(root / "a" / "doc.pdf")
            b = _touch(root / "b" / "doc.pdf")

            path = self._common(state_root, out_dir, [a, b])
            records = read_candidates(path)
            self.assertEqual([r.job_id for r in records], ["doc", "doc-1"])
            self.assertEqual(
                [Path(r.output_path).name for r in records],
                ["doc.md", "doc-1.md"],
            )
            # Outputs are absolute and under output_dir.
            for r in records:
                self.assertTrue(Path(r.output_path).is_absolute())
                self.assertEqual(Path(r.output_path).parent, out_dir.resolve())

    def test_record_carries_all_fields_with_injected_clock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_root = root / ".paperscale"
            out_dir = root / "out"
            doc = _touch(root / "paper.pdf")

            path = self._common(state_root, out_dir, [doc])
            (record,) = read_candidates(path)
            self.assertEqual(record.job_id, "paper")
            self.assertEqual(record.input_path, str(doc.resolve()))
            self.assertEqual(record.output_path, str((out_dir / "paper.md").resolve()))
            self.assertEqual(record.profile, "default")
            self.assertIsNone(record.model)
            self.assertEqual(record.base_url, "http://fake/v1")
            self.assertEqual(record.capacity, "m")
            self.assertEqual(record.created_at, 1700000000.0)

    def test_candidates_file_location_and_jsonl_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_root = root / ".paperscale"
            out_dir = root / "out"
            a = _touch(root / "a.pdf")
            b = _touch(root / "b.pdf")

            path = self._common(state_root, out_dir, [a, b])
            # Lives at candidates/<workload_id>.jsonl
            self.assertEqual(path.parent, (state_root / "candidates").resolve())
            self.assertTrue(path.name.endswith(".jsonl"))
            self.assertTrue(path.exists())

            # Every non-blank line is valid JSON.
            lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            self.assertEqual(len(lines), 2)
            for ln in lines:
                json.loads(ln)

            records = read_candidates(path)
            self.assertEqual(len(records), 2)
            # Round-trips through to_json/from_json identically.
            for rec in records:
                self.assertEqual(CandidateRecord.from_json(rec.to_json()), rec)

    def test_determinism_same_inputs_and_clock_byte_stable_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_root = root / ".paperscale"
            out_dir = root / "out"
            a = _touch(root / "a.pdf")
            b = _touch(root / "b.pdf")

            path1 = self._common(state_root, out_dir, [a, b])
            path2 = self._common(state_root, out_dir, [a, b])
            # Filenames differ (uuid) but content (records) is identical.
            self.assertNotEqual(path1.name, path2.name)
            recs1 = read_candidates(path1)
            recs2 = read_candidates(path2)
            self.assertEqual(recs1, recs2)
            # Shared created_at across all records in one manifest.
            self.assertEqual({r.created_at for r in recs1}, {1700000000.0})

    def test_no_leftover_tmp_files_after_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_root = root / ".paperscale"
            out_dir = root / "out"
            doc = _touch(root / "doc.pdf")

            path = self._common(state_root, out_dir, [doc])
            tmps = list(path.parent.glob("*.tmp"))
            self.assertEqual(tmps, [])


if __name__ == "__main__":
    unittest.main()
