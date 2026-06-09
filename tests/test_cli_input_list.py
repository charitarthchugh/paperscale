from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def _one_page_pdf(text: str) -> bytes:
    objects: list[tuple[int, str]] = []
    stream = f"BT /F1 24 Tf 72 720 Td ({text}) Tj ET"
    objects.append((1, "<< /Type /Catalog /Pages 2 0 R >>"))
    objects.append((2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>"))
    objects.append((3, "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 6 0 R >>"))
    objects.append((5, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"))
    objects.append((6, f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream"))
    output = b"%PDF-1.4\n"
    offsets = {0: 0}
    for number, body in objects:
        offsets[number] = len(output)
        output += f"{number} 0 obj\n{body}\nendobj\n".encode()
    xref_start = len(output)
    highest = max(n for n, _ in objects)
    output += f"xref\n0 {highest + 1}\n".encode()
    for number in range(0, highest + 1):
        if number in offsets and number != 0:
            output += f"{offsets[number]:010d} 00000 n \n".encode()
        else:
            output += b"0000000000 65535 f \n"
    output += f"trailer\n<< /Size {highest + 1} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n".encode()
    return output


def _enqueue_status(state_root: Path, job_id: str) -> dict:
    return json.loads((state_root / "jobs" / job_id / "indexes" / "status.json").read_text())


class CliInputListTests(unittest.TestCase):
    def _make_pdfs(self, root: Path, n: int) -> list[Path]:
        paths = []
        for i in range(n):
            p = root / f"doc{i}.pdf"
            p.write_bytes(_one_page_pdf(f"Doc{i}"))
            paths.append(p)
        return paths

    def test_enqueue_input_list_registers_one_job_per_line(self) -> None:
        from paperscale.cli import main as paperscale_main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_root = root / ".paperscale"
            pdfs = self._make_pdfs(root, 3)
            list_file = root / "files.txt"
            list_file.write_text("\n".join(str(p) for p in pdfs) + "\n", encoding="utf-8")
            out_dir = root / "out"

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = paperscale_main([
                    "enqueue", "--input-list", str(list_file), "--output-dir", str(out_dir),
                    "--state-root", str(state_root), "--base-url", "http://fake/v1", "--model", "m",
                ])
            self.assertEqual(code, 0)
            for i in range(3):
                status = _enqueue_status(state_root, f"doc{i}")
                self.assertEqual(status["pending"], 1)
                self.assertEqual(status["succeeded"], 0)
            self.assertIn("enqueued 3", stdout.getvalue())

    def test_enqueue_input_list_from_stdin_and_skips_blank_and_comment_lines(self) -> None:
        from paperscale.cli import main as paperscale_main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_root = root / ".paperscale"
            pdfs = self._make_pdfs(root, 2)
            payload = f"# a comment\n{pdfs[0]}\n\n{pdfs[1]}\n"
            out_dir = root / "out"

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout), mock.patch("sys.stdin", io.StringIO(payload)):
                code = paperscale_main([
                    "enqueue", "--input-list", "-", "--output-dir", str(out_dir),
                    "--state-root", str(state_root), "--base-url", "http://fake/v1", "--model", "m",
                ])
            self.assertEqual(code, 0)
            self.assertTrue((state_root / "jobs" / "doc0").exists())
            self.assertTrue((state_root / "jobs" / "doc1").exists())

    def test_enqueue_input_list_skips_unreadable_pdf_without_aborting_batch(self) -> None:
        from paperscale.cli import main as paperscale_main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_root = root / ".paperscale"
            good = root / "good.pdf"
            good.write_bytes(_one_page_pdf("Good"))
            bad = root / "bad.pdf"
            bad.write_bytes(b"not a pdf at all")
            list_file = root / "files.txt"
            list_file.write_text(f"{good}\n{bad}\n", encoding="utf-8")
            out_dir = root / "out"

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = paperscale_main([
                    "enqueue", "--input-list", str(list_file), "--output-dir", str(out_dir),
                    "--state-root", str(state_root), "--base-url", "http://fake/v1", "--model", "m",
                ])
            # one good job enqueued, one skipped; batch did not abort.
            self.assertTrue((state_root / "jobs" / "good").exists())
            self.assertFalse((state_root / "jobs" / "bad").exists())
            self.assertIn("skipped", stdout.getvalue().lower())
            self.assertEqual(code, 0)

    def test_enqueue_requires_input_or_input_list(self) -> None:
        from paperscale.cli import main as paperscale_main

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                paperscale_main([
                    "enqueue", "--state-root", str(Path(tmp) / ".paperscale"),
                    "--base-url", "http://fake/v1", "--model", "m",
                ])


if __name__ == "__main__":
    unittest.main()
