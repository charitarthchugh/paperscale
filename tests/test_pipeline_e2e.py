"""End-to-end worker test against a fake OpenAI-compatible server.

Exercises the full model-agnostic path: render a real PDF page, POST it to a
canned server, parse Markdown, and assemble the workspace outputs. Also covers
resume-by-skip and the ``--no-resume`` wipe.

Requires poppler (pdftoppm/pdfinfo) on PATH, like the real pipeline.
"""

import asyncio
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from pypdf import PdfWriter

from paperscale import pipeline
from paperscale.models import build_ocr_model
from paperscale.pipeline import _build_arg_parser
from paperscale.work_queue import LocalBackend, WorkQueue

CANNED_MARKDOWN = "# Canned Page\n\nHello from the fake server."

_HAVE_POPPLER = shutil.which("pdftoppm") is not None and shutil.which("pdfinfo") is not None


def _make_pdf(path: Path, pages: int = 1) -> None:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    with open(path, "wb") as f:
        writer.write(f)


class _FakeServer:
    """Minimal asyncio server returning canned chat-completion responses.

    Pass a single ``content`` string, or a ``contents`` list to return a
    different body per request (the last entry repeats once exhausted).
    """

    def __init__(self, content: str = CANNED_MARKDOWN, contents: list[str] | None = None):
        self.contents = contents if contents is not None else [content]
        self.request_count = 0
        self._server = None

    def _next_content(self) -> str:
        return self.contents[min(self.request_count - 1, len(self.contents) - 1)]

    async def __aenter__(self):
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        return self

    async def __aexit__(self, *exc):
        self._server.close()
        await self._server.wait_closed()

    @property
    def url(self) -> str:
        port = self._server.sockets[0].getsockname()[1]
        return f"http://127.0.0.1:{port}/v1"

    async def _handle(self, reader, writer):
        self.request_count += 1
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = await reader.read(4096)
            if not chunk:
                break
            data += chunk
        header_blob, _, rest = data.partition(b"\r\n\r\n")
        content_length = 0
        for line in header_blob.split(b"\r\n"):
            if line.lower().startswith(b"content-length:"):
                content_length = int(line.split(b":", 1)[1].strip())
        body = rest
        while len(body) < content_length:
            chunk = await reader.read(4096)
            if not chunk:
                break
            body += chunk

        payload = json.dumps(
            {
                "choices": [{"finish_reason": "stop", "message": {"content": self._next_content()}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            }
        ).encode()
        response = (
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: " + str(len(payload)).encode() + b"\r\n"
            b"Connection: close\r\n\r\n" + payload
        )
        writer.write(response)
        await writer.drain()
        writer.close()


def _make_args(workspace: str, server_url: str, pdf_path: str):
    parser = _build_arg_parser()
    args, _ = parser.parse_known_args(
        [workspace, "--server", server_url, "--workers", "1", "--max_concurrent_requests", "4", "--markdown"]
    )
    args.ocr_model = build_ocr_model(args.ocr_model_name)
    args.model = args.ocr_model.default_model_name
    args.target_longest_image_dim = pipeline.resolve_render_dim(args)
    return args


@unittest.skipUnless(_HAVE_POPPLER, "poppler (pdftoppm/pdfinfo) not installed")
class WorkerEndToEndTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = self._tmp.name
        self.pdf_path = str(Path(self.workspace) / "doc.pdf")
        _make_pdf(Path(self.pdf_path))

    async def asyncTearDown(self):
        self._tmp.cleanup()

    async def _run_one_work_item(self, args, queue: WorkQueue):
        await queue.initialize_queue()
        await pipeline.worker(args, queue, worker_id=0)

    async def test_full_run_writes_results_and_markdown(self):
        async with _FakeServer() as server:
            args = _make_args(self.workspace, server.url, self.pdf_path)
            queue = WorkQueue(LocalBackend(self.workspace))
            await queue.populate_queue([self.pdf_path], items_per_group=1)
            await self._run_one_work_item(args, queue)

            self.assertEqual(server.request_count, 1)

            results = list(Path(self.workspace).glob("results/output_*.jsonl"))
            self.assertEqual(len(results), 1)
            doc = json.loads(results[0].read_text().strip())
            self.assertEqual(doc["text"], CANNED_MARKDOWN)
            self.assertEqual(doc["source"], "paperscale")

            markdown_files = list(Path(self.workspace).glob("markdown/**/*.md"))
            self.assertEqual(len(markdown_files), 1)
            self.assertEqual(markdown_files[0].read_text(), CANNED_MARKDOWN)

    async def test_resume_skips_completed_work(self):
        async with _FakeServer() as server:
            args = _make_args(self.workspace, server.url, self.pdf_path)
            queue = WorkQueue(LocalBackend(self.workspace))
            await queue.populate_queue([self.pdf_path], items_per_group=1)
            await self._run_one_work_item(args, queue)
            self.assertEqual(server.request_count, 1)

            # Second pass over the same workspace: the done flag means no new work.
            resumed = WorkQueue(LocalBackend(self.workspace))
            remaining = await resumed.initialize_queue()
            self.assertEqual(remaining, 0)
            await pipeline.worker(args, resumed, worker_id=0)
            self.assertEqual(server.request_count, 1)  # no extra server calls

    async def test_no_resume_wipe_reprocesses(self):
        async with _FakeServer() as server:
            args = _make_args(self.workspace, server.url, self.pdf_path)
            queue = WorkQueue(LocalBackend(self.workspace))
            await queue.populate_queue([self.pdf_path], items_per_group=1)
            await self._run_one_work_item(args, queue)
            self.assertEqual(server.request_count, 1)

            # --no-resume wipes prior progress; re-populating yields fresh work.
            pipeline._wipe_workspace_progress(self.workspace)
            fresh = WorkQueue(LocalBackend(self.workspace))
            await fresh.populate_queue([self.pdf_path], items_per_group=1)
            self.assertEqual(await fresh.initialize_queue(), 1)
            await pipeline.worker(args, fresh, worker_id=0)
            self.assertEqual(server.request_count, 2)


@unittest.skipUnless(_HAVE_POPPLER, "poppler (pdftoppm/pdfinfo) not installed")
class ProcessPageQualityGateTests(unittest.IsolatedAsyncioTestCase):
    """The page is accepted/retried/failed by the deterministic quality verifier."""

    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = self._tmp.name
        self.pdf_path = str(Path(self.workspace) / "doc.pdf")
        _make_pdf(Path(self.pdf_path))

    async def asyncTearDown(self):
        self._tmp.cleanup()

    async def _process(self, server):
        args = _make_args(self.workspace, server.url, self.pdf_path)
        return await pipeline.process_page(args, 0, self.pdf_path, self.pdf_path, page_num=1)

    async def test_accepted_page_succeeds_first_try(self):
        async with _FakeServer(CANNED_MARKDOWN) as server:
            result = await self._process(server)
            self.assertTrue(result.is_valid)
            self.assertFalse(result.is_fallback)
            self.assertEqual(result.response.natural_text, CANNED_MARKDOWN)
            self.assertEqual(server.request_count, 1)

    async def test_refusal_is_terminal_and_not_retried(self):
        async with _FakeServer("I'm sorry, but I cannot help with that.") as server:
            result = await self._process(server)
            # Terminal verdict short-circuits retries and falls back to pdftotext.
            self.assertTrue(result.is_fallback)
            self.assertEqual(server.request_count, 1)

    async def test_blank_page_empty_output_accepted(self):
        # Blank render + empty model output -> accepted as a blank page (no retry).
        async with _FakeServer("") as server:
            result = await self._process(server)
            self.assertTrue(result.is_valid)
            self.assertFalse(result.is_fallback)
            self.assertIsNone(result.response.natural_text)
            self.assertEqual(server.request_count, 1)

    async def test_retryable_failure_then_success(self):
        # Mojibake is retryable (and not blank-eligible), so the page retries and
        # then succeeds on the second, clean response.
        async with _FakeServer(contents=["garbled ���� output", CANNED_MARKDOWN]) as server:
            result = await self._process(server)
            self.assertTrue(result.is_valid)
            self.assertFalse(result.is_fallback)
            self.assertEqual(result.response.natural_text, CANNED_MARKDOWN)
            self.assertEqual(server.request_count, 2)


class BuildPageQueryTests(unittest.TestCase):
    """build_page_query merges the adapter's sampling params, temperature aside."""

    def test_lightonocr2_includes_top_p(self):
        query = pipeline.build_page_query("IMG", build_ocr_model("lightonocr2"), "served")
        self.assertEqual(query["model"], "served")
        self.assertEqual(query["top_p"], 0.9)

    def test_markdown_has_no_top_p(self):
        query = pipeline.build_page_query("IMG", build_ocr_model("markdown"), "served")
        self.assertNotIn("top_p", query)

    def test_temperature_remains_caller_controlled(self):
        query = pipeline.build_page_query("IMG", build_ocr_model("lightonocr2"), "served")
        query["temperature"] = 0.5  # the per-attempt override the caller applies
        self.assertEqual(query["temperature"], 0.5)


class ResolveRenderDimTests(unittest.TestCase):
    """--target_longest_image_dim defaults to the model's preferred when omitted."""

    def _args(self, extra):
        parser = _build_arg_parser()
        args, _ = parser.parse_known_args(["ws", *extra])
        args.ocr_model = build_ocr_model(args.ocr_model_name)
        return args

    def test_model_preferred_when_flag_absent(self):
        args = self._args(["--ocr-model", "lightonocr2"])
        self.assertIsNone(args.target_longest_image_dim)  # parser default
        self.assertEqual(pipeline.resolve_render_dim(args), 1540)

    def test_markdown_default_dim(self):
        args = self._args([])
        self.assertEqual(pipeline.resolve_render_dim(args), 1288)

    def test_explicit_flag_wins(self):
        args = self._args(["--ocr-model", "lightonocr2", "--target_longest_image_dim", "999"])
        self.assertEqual(pipeline.resolve_render_dim(args), 999)


if __name__ == "__main__":
    unittest.main()
