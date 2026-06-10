"""paperscale OCR pipeline.

A local, model-agnostic reimplementation of olmOCR's batch pipeline: a work
queue of PDF/image groups is drained by a pool of async workers that render each
page, send it to an OpenAI-compatible OCR model, and assemble per-document
Dolma JSONL (plus optional Markdown) into the workspace.

Three deliberate departures from olmOCR:

* **Decoupled models** — the prompt and response parsing live in an
  :class:`~paperscale.models.base.OCRModel` adapter (``--ocr-model``), so any
  model that emits Markdown works, not just olmOCR.
* **Deterministic quality gate** — page acceptance is decided by paperscale's
  :class:`~paperscale.quality.verifier.DeterministicQualityVerifier` (empty /
  mojibake / control-char / refusal / repetition / truncation / length checks),
  not olmOCR's token-count, finish_reason, and rotation heuristics. Refusals are
  terminal (no retry); near-blank pages that OCR to empty are accepted as blank.
* **Opt-out resume** — completed work items are skipped on restart by default
  (olmOCR's done-flag behavior); ``--no-resume`` wipes prior progress and
  reprocesses the workspace from scratch.

S3, beaker, and olmOCR's page-rotation protocol are intentionally omitted.
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import base64
import datetime
import errno
import glob
import hashlib
import json
import logging
import multiprocessing
import os
import random
import re
import shutil
import ssl
import sys
import tarfile
import tempfile
from dataclasses import dataclass, replace
from functools import cache
from urllib.parse import urlparse

import httpx
from pypdf import PdfReader
from tqdm import tqdm

from paperscale.anchor import get_anchor_text
from paperscale.check import check_poppler_version, check_torch_gpu_available
from paperscale.filter import Language, PdfFilter
from paperscale.image_utils import convert_image_to_pdf_bytes, is_jpeg, is_png
from paperscale.metrics import MetricsKeeper, WorkerTracker
from paperscale.models import DEFAULT_MODEL, MODEL_REGISTRY, OCRModel, build_ocr_model
from paperscale.prompts import PageResponse
from paperscale.quality.verifier import DeterministicQualityVerifier
from paperscale.renderpdf import png_dark_fraction, render_pdf_to_base64png
from paperscale.version import VERSION
from paperscale.work_queue import DONE_FLAGS_DIR, WORKER_LOCKS_DIR, LocalBackend, WorkQueue

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

server_logger = logging.getLogger("vllm")
server_logger.propagate = False

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

logger.addHandler(console_handler)
server_logger.addHandler(console_handler)

# Quiet logs from pypdf
logging.getLogger("pypdf").setLevel(logging.ERROR)

# Global token statistics
metrics = MetricsKeeper(window=60 * 5)
tracker = WorkerTracker()

# Updated by vllm_server_task; lets process_page fire parallel retries when idle.
vllm_queued_requests: int | None = None

# Higher temperature on later retries helps overcome repetition issues.
TEMPERATURE_BY_ATTEMPT = [0.1, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 1.0]

pdf_render_max_workers_limit = asyncio.BoundedSemaphore(max(1, multiprocessing.cpu_count() - 2))
max_concurrent_requests_limit = asyncio.BoundedSemaphore(1)  # Actual value set by args in main()

# Filter object, cached so it only loads if/when --apply_filter is used.
get_pdf_filter = cache(lambda: PdfFilter(languages_to_keep={Language.ENGLISH, None}, apply_download_spam_check=True, apply_form_check=True))

# Page acceptance uses paperscale's deterministic quality gate (empty / mojibake /
# control-chars / refusal / repetition / truncation / length checks) instead of
# olmOCR's token-count + finish_reason + rotation heuristics.
_verifier = DeterministicQualityVerifier()

# A page whose rendered image has less ink than this is treated as genuinely blank:
# an empty/degenerate OCR result for it is accepted as a successful empty page
# rather than retried. Blank scans sit ~0.002-0.005; content pages are higher.
_BLANK_INK_THRESHOLD = 0.01

# Quality diagnostics that, on a near-blank render, mean "genuinely blank page"
# (empty output, or a model repetition loop on noise) rather than a read failure.
_BLANK_ELIGIBLE_DIAGNOSTICS = frozenset({"empty_output", "repeated_ngram", "repeated_character"})


def _render_is_blank(image_base64: str) -> bool:
    try:
        return png_dark_fraction(base64.b64decode(image_base64)) < _BLANK_INK_THRESHOLD
    except Exception:  # never let a blank-check error fail a page
        return False


@dataclass(frozen=True)
class PageResult:
    source_path: str
    page_num: int
    response: PageResponse

    input_tokens: int
    output_tokens: int
    is_fallback: bool
    # Page passed the deterministic quality gate (or was accepted as a blank page).
    is_valid: bool
    # Quality verdict was a terminal failure (e.g. a model refusal): do not retry.
    is_terminal: bool = False


def build_page_query(image_base64: str, ocr_model: OCRModel, served_model_name: str) -> dict:
    MAX_TOKENS = 8000
    return {
        "model": served_model_name,
        "messages": ocr_model.build_messages(image_base64),
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,  # Overridden per attempt by the caller.
    }


async def try_single_page(
    args,
    pdf_orig_path: str,
    page_num: int,
    attempt: int,
    image_base64: str,
    render_is_blank: bool,
) -> PageResult | None:
    """Try processing a single page once. Returns PageResult on success, None on failure.

    Acceptance is decided by paperscale's deterministic quality verifier on the
    parsed Markdown (not olmOCR's token-count/finish_reason/rotation heuristics):
    accepted pages succeed, model refusals are terminal, everything else is
    retryable, and an empty/degenerate result on a near-blank render is accepted
    as a genuinely blank page.

    Does NOT handle retries - caller is responsible for retry logic.
    """
    COMPLETION_URL = f"{args.server.rstrip('/')}/chat/completions"

    temp_idx = min(attempt, len(TEMPERATURE_BY_ATTEMPT) - 1)
    temperature = TEMPERATURE_BY_ATTEMPT[temp_idx]

    api_key = args.api_key if args.server and hasattr(args, "api_key") else None

    try:
        query = build_page_query(image_base64, args.ocr_model, args.model)
        query["temperature"] = temperature

        if args.guided_decoding and (guided_regex := args.ocr_model.guided_regex()):
            query["guided_regex"] = guided_regex

        async with max_concurrent_requests_limit:
            status_code, response_body = await apost(COMPLETION_URL, json_data=query, api_key=api_key)

        if status_code != 200:
            logger.warning(
                f"Server returned {status_code} for {pdf_orig_path}-{page_num} attempt {attempt}: {response_body[:500] if response_body else 'empty response'}"
            )
            return None

        base_response_data = json.loads(response_body)

        metrics.add_metrics(
            server_input_tokens=base_response_data["usage"].get("prompt_tokens", 0),
            server_output_tokens=base_response_data["usage"].get("completion_tokens", 0),
        )

        model_response_markdown = base_response_data["choices"][0]["message"]["content"]
        page_response = args.ocr_model.parse(model_response_markdown)

        finding = _verifier.classify(page_response.natural_text or "")
        if finding.accepted:
            is_valid, is_terminal = True, False
        elif finding.kind in _BLANK_ELIGIBLE_DIAGNOSTICS and render_is_blank:
            # Genuinely blank page: accept it as an empty page rather than retry.
            page_response = replace(page_response, natural_text=None)
            is_valid, is_terminal = True, False
            metrics.add_metrics(blank_pages=1)
        else:
            is_valid = False
            is_terminal = finding.retry_class == "terminal"
            metrics.add_metrics(**{f"quality_reject_{finding.kind}": 1})

        return PageResult(
            pdf_orig_path,
            page_num,
            page_response,
            input_tokens=base_response_data["usage"].get("prompt_tokens", 0),
            output_tokens=base_response_data["usage"].get("completion_tokens", 0),
            is_fallback=False,
            is_valid=is_valid,
            is_terminal=is_terminal,
        )
    except asyncio.CancelledError:
        raise
    except (ConnectionError, OSError, asyncio.TimeoutError):
        # Re-raise connection errors so caller can apply exponential backoff
        raise
    except Exception as e:
        logger.warning(f"try_single_page failed for {pdf_orig_path}-{page_num} attempt {attempt}: {type(e).__name__}: {e}")
        return None


def make_fallback_result(pdf_orig_path: str, pdf_local_path: str, page_num: int) -> PageResult:
    """Create a fallback PageResult using pdftotext."""
    return PageResult(
        pdf_orig_path,
        page_num,
        PageResponse(
            natural_text=get_anchor_text(pdf_local_path, page_num, pdf_engine="pdftotext"),
            primary_language=None,
            is_rotation_valid=True,
            rotation_correction=0,
            is_table=False,
            is_diagram=False,
        ),
        input_tokens=0,
        output_tokens=0,
        is_fallback=True,
        is_valid=True,
    )


async def try_single_page_with_backoff(
    args,
    pdf_orig_path: str,
    page_num: int,
    attempt: int,
    image_base64: str,
    render_is_blank: bool,
) -> PageResult | None:
    """Wrapper around try_single_page that retries transient send-side errors.

    File-descriptor exhaustion on the sending machine (EMFILE/ENFILE) is a soft
    drop: the request is backed off and retried indefinitely (capped delay)
    rather than counted as a failure, so it never consumes one of the page's
    retries and never aborts the job. It self-resolves as in-flight requests
    release their sockets. Other connection errors use a bounded exponential
    backoff and abort the job only if they never recover.
    """
    MAX_BACKOFF_ATTEMPTS = 10
    FD_MAX_BACKOFF_SECONDS = 30

    conn_backoff = 0
    fd_backoff = 0
    while True:
        try:
            return await try_single_page(args, pdf_orig_path, page_num, attempt, image_base64, render_is_blank)
        except OSError as e:
            if e.errno in (errno.EMFILE, errno.ENFILE):
                # Out of file descriptors on this machine: drop the request for now
                # without failing it (does not consume a page retry).
                sleep_delay = min(2**fd_backoff, FD_MAX_BACKOFF_SECONDS)
                fd_backoff += 1
                metrics.add_metrics(fd_exhaustion_drops=1)
                logger.warning(
                    f"Out of file descriptors sending {pdf_orig_path}-{page_num} (errno {e.errno}); "
                    f"dropping request without failing it, retrying in {sleep_delay}s"
                )
                await asyncio.sleep(sleep_delay)
                continue

            conn_backoff += 1
            if conn_backoff > MAX_BACKOFF_ATTEMPTS:
                logger.error(f"Max backoff attempts reached for {pdf_orig_path}-{page_num}, terminating job")
                sys.exit(1)
            sleep_delay = 10 * (2 ** (conn_backoff - 1))
            logger.warning(
                f"Connection error on {pdf_orig_path}-{page_num} attempt {attempt}: {type(e).__name__}: {e}. "
                f"Backoff {conn_backoff}/{MAX_BACKOFF_ATTEMPTS}, sleeping {sleep_delay}s"
            )
            await asyncio.sleep(sleep_delay)


async def process_page(args, worker_id: int, pdf_orig_path: str, pdf_local_path: str, page_num: int) -> PageResult:
    """Process a single page, retrying until the quality verifier accepts it.

    There is no rotation protocol: the page is rendered once and reused across
    attempts. A page is done when it passes the deterministic quality gate; a
    terminal verdict (model refusal) stops retrying immediately; otherwise we
    retry up to ``max_page_retries`` and finally fall back to pdftotext.
    """
    MAX_RETRIES = args.max_page_retries

    await tracker.track_work(worker_id, f"{pdf_orig_path}-{page_num}", "started")

    # Render the page once; the same image is sent on every retry attempt.
    try:
        async with pdf_render_max_workers_limit:
            image_base64 = await asyncio.to_thread(
                render_pdf_to_base64png, pdf_local_path, page_num, target_longest_image_dim=args.target_longest_image_dim
            )
    except Exception:
        logger.exception(f"Failed to render {pdf_orig_path}-{page_num}, using fallback")
        metrics.add_metrics(failed_pages=1)
        await tracker.track_work(worker_id, f"{pdf_orig_path}-{page_num}", "errored")
        return make_fallback_result(pdf_orig_path, pdf_local_path, page_num)

    render_is_blank = _render_is_blank(image_base64)

    for i in range(MAX_RETRIES):
        result = await try_single_page_with_backoff(args, pdf_orig_path, page_num, i, image_base64, render_is_blank)

        if result is not None and result.is_valid:
            metrics.add_metrics(**{"completed_pages": 1, f"finished_on_attempt_{i}": 1})
            await tracker.track_work(worker_id, f"{pdf_orig_path}-{page_num}", "finished")
            return result

        # A terminal quality verdict (e.g. a refusal) will not improve on retry.
        if result is not None and result.is_terminal:
            logger.error(f"Terminal quality failure for {pdf_orig_path}-{page_num} ({result.response.natural_text is None})")
            break

        # When the server queue is idle, fire all remaining retries in parallel.
        remaining = list(range(i + 1, MAX_RETRIES))
        if remaining and vllm_queued_requests == 0:
            logger.info(f"Queue empty, firing {len(remaining)} parallel retries for {pdf_orig_path}-{page_num}")
            tasks = [
                asyncio.create_task(try_single_page_with_backoff(args, pdf_orig_path, page_num, a, image_base64, render_is_blank))
                for a in remaining
            ]
            for coro in asyncio.as_completed(tasks):
                try:
                    parallel_result = await coro
                except asyncio.CancelledError:
                    continue
                if parallel_result is not None and parallel_result.is_valid:
                    for t in tasks:
                        t.cancel()
                    metrics.add_metrics(**{"completed_pages": 1, "finished_on_parallel_retry": 1})
                    await tracker.track_work(worker_id, f"{pdf_orig_path}-{page_num}", "finished")
                    return parallel_result
            break  # Parallel attempts exhausted

    logger.error(f"Failed {pdf_orig_path}-{page_num} after {MAX_RETRIES} attempts")
    metrics.add_metrics(failed_pages=1)
    await tracker.track_work(worker_id, f"{pdf_orig_path}-{page_num}", "errored")
    return make_fallback_result(pdf_orig_path, pdf_local_path, page_num)


# Manual simple implementation of HTTP Post.
# httpx and aiohttp are very complex beasts; at the scale of 100M+ requests their
# connection pools deadlock in strange ways, so olmOCR speaks HTTP directly.
async def apost(url, json_data, api_key=None):
    parsed_url = urlparse(url)
    host = parsed_url.hostname
    if parsed_url.scheme == "https":
        port = parsed_url.port or 443
        use_ssl = True
    else:
        port = parsed_url.port or 80
        use_ssl = False
    path = parsed_url.path or "/"

    writer = None
    try:
        if use_ssl:
            ssl_context = ssl.create_default_context()
            reader, writer = await asyncio.open_connection(host, port, ssl=ssl_context)
        else:
            reader, writer = await asyncio.open_connection(host, port)

        json_payload = json.dumps(json_data)

        headers = [
            f"POST {path} HTTP/1.1",
            f"Host: {host}",
            "Content-Type: application/json",
            f"Content-Length: {len(json_payload)}",
        ]

        if api_key:
            headers.append(f"Authorization: Bearer {api_key}")

        headers.append("Connection: close")

        request = "\r\n".join(headers) + "\r\n\r\n" + json_payload
        writer.write(request.encode())
        await writer.drain()

        status_line = await reader.readline()
        if not status_line:
            raise ConnectionError("No response from server")
        status_parts = status_line.decode().strip().split(" ", 2)
        if len(status_parts) < 2:
            raise ValueError(f"Malformed status line: {status_line.decode().strip()}")
        status_code = int(status_parts[1])

        headers = {}
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break
            key, _, value = line.decode().partition(":")
            headers[key.strip().lower()] = value.strip()

        if "content-length" in headers:
            body_length = int(headers["content-length"])
            response_body = await reader.readexactly(body_length)
        elif headers.get("transfer-encoding", "") == "chunked":
            chunks = []
            while True:
                size_line = await reader.readline()
                chunk_size = int(size_line.strip(), 16)

                if chunk_size == 0:
                    await reader.readline()
                    break

                chunk_data = await reader.readexactly(chunk_size)
                chunks.append(chunk_data)
                await reader.readline()

            response_body = b"".join(chunks)
        elif headers.get("connection", "") == "close":
            response_body = await reader.read()
        else:
            raise ConnectionError("Cannot determine response body length")

        return status_code, response_body
    finally:
        if writer is not None:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass


def is_tarball_path(path: str) -> bool:
    """Check if a path is a tarball based on extension."""
    lower = path.lower()
    return lower.endswith(".tar.gz") or lower.endswith(".tgz")


async def process_tarball(args, worker_id: int, tarball_path: str) -> list:
    """Process all PDFs inside a local tarball concurrently and return Dolma documents."""
    logger.info(f"Worker {worker_id} processing tarball {tarball_path}")

    temp_dir = tempfile.mkdtemp()
    try:
        pdf_files = []  # (source_path, local_path)
        with tarfile.open(tarball_path, mode="r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.lower().endswith(".pdf"):
                    local_path = os.path.join(temp_dir, os.path.basename(member.name))
                    extracted = tar.extractfile(member)
                    if extracted:
                        with open(local_path, "wb") as f:
                            f.write(extracted.read())
                        pdf_files.append((f"{tarball_path}::{member.name}", local_path))

        logger.info(f"Worker {worker_id} extracted {len(pdf_files)} PDFs from {tarball_path}")

        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(process_single_pdf(args, worker_id, src, local)) for src, local in pdf_files]

        dolma_docs = [t.result() for t in tasks if t.result() is not None]
        logger.info(f"Worker {worker_id} processed {len(dolma_docs)} PDFs from tarball {tarball_path}")
        return dolma_docs
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


async def process_single_pdf(args, worker_id: int, pdf_orig_path: str, local_pdf_path: str):
    """Process a single PDF that's already on disk. Returns a Dolma document or None."""
    try:
        try:
            reader = PdfReader(local_pdf_path)
            num_pages = reader.get_num_pages()
        except Exception:
            logger.exception(f"Could not count number of pages for {pdf_orig_path}, aborting document")
            return None

        logger.debug(f"Got {num_pages} pages to do for {pdf_orig_path} in worker {worker_id}")

        if args.apply_filter and get_pdf_filter().filter_out_pdf(local_pdf_path):
            logger.info(f"Filtering out pdf {pdf_orig_path}")
            return None

        page_tasks = []

        async with asyncio.TaskGroup() as tg:
            for page_num in range(1, num_pages + 1):
                task = tg.create_task(process_page(args, worker_id, pdf_orig_path, local_pdf_path, page_num))
                page_tasks.append(task)

        page_results = [task.result() for task in page_tasks]
        assert all(page_result.is_valid for page_result in page_results)

        num_fallback_pages = sum(page_result.is_fallback for page_result in page_results)

        if num_fallback_pages / num_pages > args.max_page_error_rate:
            logger.error(
                f"Document {pdf_orig_path} has {num_fallback_pages} fallback pages out of {num_pages} exceeding "
                f"max_page_error_rate of {args.max_page_error_rate}, discarding document."
            )
            return None
        elif num_fallback_pages > 0:
            logger.warning(
                f"Document {pdf_orig_path} processed with {num_fallback_pages} fallback pages out of {num_pages}, proceeding to build Dolma document."
            )

        return build_dolma_document(pdf_orig_path, page_results)
    except Exception as e:
        logger.exception(f"Exception in process_single_pdf for {pdf_orig_path}: {e}")
        return None


async def process_pdf(args, worker_id: int, pdf_orig_path: str):
    """Process a single local PDF/image path and return a Dolma document."""
    if not os.path.exists(pdf_orig_path):
        logger.info(f"File not found, skipping it completely {pdf_orig_path}")
        return None

    # Images are converted to a single-page PDF in a temp file.
    if is_png(pdf_orig_path) or is_jpeg(pdf_orig_path):
        logger.info(f"Converting {pdf_orig_path} from image to PDF format...")
        with tempfile.NamedTemporaryFile("wb+", suffix=".pdf", delete=False) as tf:
            tf.write(convert_image_to_pdf_bytes(pdf_orig_path))
            tf.flush()
            local_pdf_path = tf.name
        try:
            return await process_single_pdf(args, worker_id, pdf_orig_path, local_pdf_path)
        finally:
            if os.path.exists(local_pdf_path):
                os.unlink(local_pdf_path)

    return await process_single_pdf(args, worker_id, pdf_orig_path, pdf_orig_path)


def build_dolma_document(pdf_orig_path, page_results):
    document_text = ""
    pdf_page_spans = []
    current_char_pos = 0

    for index, page_result in enumerate(page_results):
        if page_result.response.natural_text is not None:
            content = page_result.response.natural_text + ("\n" if index < len(page_results) - 1 else "")
        else:
            content = ""

        start_pos = current_char_pos
        document_text += content
        current_char_pos = len(document_text)
        pdf_page_spans.append([start_pos, current_char_pos, page_result.page_num])

    if not document_text:
        logger.info(f"No document text for {pdf_orig_path}")
        return None

    metadata = {
        "Source-File": pdf_orig_path,
        "paperscale-version": VERSION,
        "pdf-total-pages": len(page_results),
        "total-input-tokens": sum(page.input_tokens for page in page_results),
        "total-output-tokens": sum(page.output_tokens for page in page_results),
        "total-fallback-pages": sum(page.is_fallback for page in page_results),
    }

    id_ = hashlib.sha1(document_text.encode()).hexdigest()

    return {
        "id": id_,
        "text": document_text,
        "source": "paperscale",
        "added": datetime.datetime.now().strftime("%Y-%m-%d"),
        "created": datetime.datetime.now().strftime("%Y-%m-%d"),
        "metadata": metadata,
        "attributes": {
            "pdf_page_numbers": pdf_page_spans,
            "primary_language": [p.response.primary_language for p in page_results],
            "is_rotation_valid": [p.response.is_rotation_valid for p in page_results],
            "rotation_correction": [p.response.rotation_correction for p in page_results],
            "is_table": [p.response.is_table for p in page_results],
            "is_diagram": [p.response.is_diagram for p in page_results],
        },
    }


def get_markdown_path(workspace: str, source_file: str) -> str:
    """Calculate the local markdown output path for a given source file."""
    # Handle tarball paths (format: tarball_path::internal_path)
    if "::" in source_file:
        tarball_path, internal_path = source_file.split("::", 1)
        tarball_basename = os.path.splitext(os.path.basename(tarball_path))[0]
        if tarball_basename.endswith(".tar"):
            tarball_basename = tarball_basename[:-4]
        relative_path = os.path.join(tarball_basename, internal_path)
    else:
        relative_path = source_file.lstrip("/")

    # Sanitize path: remove any .. components to prevent path traversal
    parts = relative_path.split("/")
    safe_parts = [p for p in parts if p and p != ".."]
    relative_path = "/".join(safe_parts)

    md_filename = os.path.splitext(os.path.basename(relative_path))[0] + ".md"
    dir_path = os.path.dirname(relative_path)

    return os.path.join(workspace, "markdown", dir_path, md_filename)


async def worker(args, work_queue: WorkQueue, worker_id):
    while True:
        work_item = await work_queue.get_work()

        if work_item is None:
            logger.info(f"Worker {worker_id} exiting due to empty queue")
            break

        logger.info(f"Worker {worker_id} processing work item {work_item.hash}")
        await tracker.clear_work(worker_id)

        try:
            async with asyncio.TaskGroup() as tg:
                dolma_tasks: list[asyncio.Task] = []
                for path in work_item.work_paths:
                    if is_tarball_path(path):
                        dolma_tasks.append(tg.create_task(process_tarball(args, worker_id, path)))
                    else:
                        dolma_tasks.append(tg.create_task(process_pdf(args, worker_id, path)))
                logger.info(f"Created all tasks for {work_item.hash}")

            logger.info(f"Finished TaskGroup for worker on {work_item.hash}")

            dolma_docs = []
            for task in dolma_tasks:
                try:
                    result = task.result()
                except Exception:
                    result = None

                if result is None:
                    continue
                if isinstance(result, list):  # process_tarball returns a list
                    dolma_docs.extend(result)
                else:
                    dolma_docs.append(result)

            logger.info(f"Got {len(dolma_docs)} docs for {work_item.hash}")

            output_final_path = os.path.join(args.workspace, "results", f"output_{work_item.hash}.jsonl")
            os.makedirs(os.path.dirname(output_final_path), exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w+", delete=False, dir=os.path.dirname(output_final_path)) as tf:
                for doc in dolma_docs:
                    tf.write(json.dumps(doc))
                    tf.write("\n")
                tf.flush()
                temp_path = tf.name
            os.replace(temp_path, output_final_path)

            if args.markdown:
                logger.info(f"Writing {len(dolma_docs)} markdown files for {work_item.hash}")
                for doc in dolma_docs:
                    source_file = doc["metadata"]["Source-File"]
                    markdown_path = get_markdown_path(args.workspace, source_file)
                    os.makedirs(os.path.dirname(markdown_path), exist_ok=True)
                    with open(markdown_path, "w") as md_f:
                        md_f.write(doc["text"])

            metrics.add_metrics(
                finished_input_tokens=sum(doc["metadata"]["total-input-tokens"] for doc in dolma_docs),
                finished_output_tokens=sum(doc["metadata"]["total-output-tokens"] for doc in dolma_docs),
            )

            await work_queue.mark_done(work_item)
        except Exception as e:
            logger.exception(f"Exception occurred while processing work_hash {work_item.hash}: {e}")


async def vllm_server_task(model_name_or_path, args, unknown_args=None):
    cmd = [
        "vllm",
        "serve",
        model_name_or_path,
        "--port",
        str(args.port),
        "--disable-log-requests",
        "--uvicorn-log-level",
        "warning",
        "--served-model-name",
        args.model,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--data-parallel-size",
        str(args.data_parallel_size),
        "--limit-mm-per-prompt",
        '{"video": 0}',
    ]

    if args.gpu_memory_utilization is not None:
        cmd.extend(["--gpu-memory-utilization", str(args.gpu_memory_utilization)])

    if args.max_model_len is not None:
        cmd.extend(["--max-model-len", str(args.max_model_len)])

    if unknown_args:
        cmd.extend(unknown_args)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        # OMP_NUM_THREADS=1 avoids contention if several copies run on a multi-GPU box.
        env={**os.environ, "OMP_NUM_THREADS": "1"},
    )

    def _kill_proc():
        try:
            proc.terminate()
        except Exception:
            logger.info("VLLM Process already terminated")

    atexit.register(_kill_proc)

    last_running_req, peak_running_req, last_queue_req = 0, 0, 0
    server_printed_ready_message = False

    async def process_line(line):
        nonlocal last_running_req, last_queue_req, peak_running_req, server_printed_ready_message
        server_logger.info(line)

        if "Detected errors during sampling" in line:
            logger.error("Cannot continue, sampling errors detected, model is probably corrupt")
            sys.exit(1)

        if not server_printed_ready_message and ("The server is fired up and ready to roll!" in line or "Starting vLLM API server" in line):
            server_printed_ready_message = True

        if match := re.search(r"Running: (\d+)", line):
            current_running = int(match.group(1))
            if current_running > peak_running_req:
                peak_running_req = current_running
                logger.info(f"New peak running requests: {peak_running_req}")
            last_running_req = current_running

        if match := re.search(r"(?:Waiting|Pending):\s*(\d+)", line):
            global vllm_queued_requests
            last_queue_req = int(match.group(1))
            vllm_queued_requests = last_queue_req
            logger.info(f"vllm running req: {last_running_req} queue req: {last_queue_req}")

    async def read_stream(stream):
        while True:
            line = await stream.readline()
            if not line:
                break
            try:
                line = line.decode("utf-8").rstrip()
                await process_line(line)
            except Exception as ex:
                logger.warning(f"Got {ex} when reading log line from inference server, skipping")

    stdout_task = asyncio.create_task(read_stream(proc.stdout))
    stderr_task = asyncio.create_task(read_stream(proc.stderr))

    try:
        await proc.wait()
    except asyncio.CancelledError:
        logger.info("Got cancellation request for VLLM server")
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("VLLM server did not terminate within 10 seconds")
        raise

    await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)


async def vllm_server_host(model_name_or_path, args, unknown_args=None):
    MAX_RETRIES = 5
    retry = 0

    while retry < MAX_RETRIES:
        await vllm_server_task(model_name_or_path, args, unknown_args)
        logger.warning("VLLM server task ended")
        retry += 1

    if retry >= MAX_RETRIES:
        logger.error(f"Ended up starting the vllm server more than {retry} times, cancelling pipeline")
        logger.error("Please make sure vllm is installed: https://docs.vllm.ai/en/stable/getting_started/installation/gpu.html")
        sys.exit(1)


async def vllm_server_ready(args):
    max_attempts = args.max_server_ready_timeout
    delay_sec = 1
    url = f"{args.server.rstrip('/')}/models"

    for attempt in range(1, max_attempts + 1):
        try:
            headers = {}
            if args.server and hasattr(args, "api_key") and args.api_key:
                headers["Authorization"] = f"Bearer {args.api_key}"

            async with httpx.AsyncClient() as session:
                response = await session.get(url, headers=headers)

                if response.status_code == 200:
                    logger.info("vllm server is ready.")
                    return
                else:
                    logger.info(f"Attempt {attempt}: Unexpected status code {response.status_code}")
        except Exception:
            logger.warning(f"Attempt {attempt}: Please wait for vllm server to become ready...")

        await asyncio.sleep(delay_sec)

    raise Exception("vllm server did not become ready after waiting.")


async def download_model(model_name_or_path: str, max_retries: int = 5):
    for retry in range(max_retries):
        try:
            if os.path.isabs(model_name_or_path) and os.path.isdir(model_name_or_path):
                logger.info(f"Using local model path at '{model_name_or_path}'")
                return model_name_or_path
            else:
                logger.info(f"Downloading model with hugging face '{model_name_or_path}'")
                from huggingface_hub import snapshot_download  # type: ignore

                snapshot_download(repo_id=model_name_or_path)
                return model_name_or_path
        except Exception:
            if retry == max_retries - 1:
                raise

            logger.exception(f"Could not download model, retrying ({retry + 1}/{max_retries})")
            await asyncio.sleep(random.randrange(10, 30) * 2**retry)


async def metrics_reporter(work_queue):
    while True:
        logger.info(f"Queue remaining: {work_queue.size}")
        logger.info("\n" + str(metrics))
        logger.info("\n" + str(await tracker.get_status_table()))
        await asyncio.sleep(10)


def print_stats(args):
    """Report progress + token statistics for a local workspace."""
    LONG_CONTEXT_THRESHOLD = 32768

    done_work_items = glob.glob(os.path.join(args.workspace, DONE_FLAGS_DIR, "done_*.flag"))
    result_files = glob.glob(os.path.join(args.workspace, "results", "*.jsonl"))

    totals = {"docs": 0, "input_tokens": 0, "output_tokens": 0, "pages": 0, "fallback_pages": 0, "long_docs": 0, "long_tokens": 0}
    for result_path in result_files:
        try:
            with open(result_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    doc = json.loads(line)
                    meta = doc["metadata"]
                    out_tokens = meta.get("total-output-tokens", 0)
                    totals["docs"] += 1
                    totals["input_tokens"] += meta.get("total-input-tokens", 0)
                    totals["output_tokens"] += out_tokens
                    totals["pages"] += meta.get("pdf-total-pages", 0)
                    totals["fallback_pages"] += meta.get("total-fallback-pages", 0)
                    if out_tokens > LONG_CONTEXT_THRESHOLD:
                        totals["long_docs"] += 1
                        totals["long_tokens"] += out_tokens
        except Exception as e:
            logger.warning(f"Error processing {result_path}: {e}")

    d, p, o = totals["docs"], totals["pages"], totals["output_tokens"]
    print(
        f"""
Work Items Status:
Completed work items: {len(done_work_items):,}
Result files: {len(result_files):,}

Results:
Total documents processed: {d:,}
Total pages on fallback: {totals['fallback_pages']:,}
Total pages processed: {p:,}

Total input tokens: {totals['input_tokens']:,}
Total output tokens: {o:,}

Average pages per doc: {p / max(1, d):,.1f}
Average output tokens per doc: {o / max(1, d):,.1f}
Average output tokens per page: {o / max(1, p):,.1f}

Long Context Documents (>{LONG_CONTEXT_THRESHOLD} tokens): {totals['long_docs']:,}
Total tokens in long context documents: {totals['long_tokens']:,}"""
    )


def _expand_pdf_inputs(pdf_args: list[str]) -> tuple[set[str], set[str]]:
    """Expand --pdfs into (pdf/image paths, tarball paths). Local paths and globs only."""
    pdf_work_paths: set[str] = set()
    tarball_paths: set[str] = set()

    for pdf_path in pdf_args:
        # Expand local glob patterns first.
        matches = glob.glob(pdf_path) if any(ch in pdf_path for ch in "*?[") else [pdf_path]
        if not matches:
            logger.warning(f"No files matched {pdf_path}")
        for match in matches:
            if not os.path.exists(match):
                raise ValueError(f"pdfs argument needs to be a local path or glob; not found: {match}")
            if is_tarball_path(match):
                tarball_paths.add(match)
            elif match.lower().endswith((".pdf", ".png", ".jpg", ".jpeg")):
                with open(match, "rb") as f:
                    head = f.read(4)
                if head == b"%PDF" or is_png(match) or is_jpeg(match):
                    pdf_work_paths.add(match)
                else:
                    logger.warning(f"File at {match} is not a valid PDF/image, skipping")
            elif match.lower().endswith(".txt"):
                logger.info(f"Loading file at {match} as list of paths")
                with open(match) as f:
                    lines = [line.strip() for line in f if line.strip()]
                tarball_paths.update(p for p in lines if is_tarball_path(p))
                pdf_work_paths.update(p for p in lines if not is_tarball_path(p))
            else:
                raise ValueError(f"Unsupported file extension for {match}")

    return pdf_work_paths, tarball_paths


def _estimate_items_per_group(pdf_work_paths: set[str], pages_per_group: int) -> int:
    sample_size = min(100, len(pdf_work_paths))
    sampled_pdfs = random.sample(list(pdf_work_paths), sample_size)
    page_counts = []

    for pdf in tqdm(sampled_pdfs, desc="Sampling PDFs to calculate optimal length"):
        try:
            if is_png(pdf) or is_jpeg(pdf):
                page_counts.append(1)
            else:
                page_counts.append(len(PdfReader(pdf).pages))
        except Exception as e:
            logger.warning(f"Failed to read {pdf}: {e}")

    avg_pages_per_pdf = (sum(page_counts) / len(page_counts)) if page_counts else 10
    items_per_group = max(1, int(pages_per_group / avg_pages_per_pdf))
    logger.info(f"Calculated items_per_group: {items_per_group} based on average pages per PDF: {avg_pages_per_pdf:.2f}")
    return items_per_group


def _wipe_workspace_progress(workspace: str) -> None:
    """--no-resume: drop prior done flags, results, and worker locks."""
    for sub in ("results", DONE_FLAGS_DIR, WORKER_LOCKS_DIR):
        shutil.rmtree(os.path.join(workspace, sub), ignore_errors=True)
    logger.info("Cleared prior workspace progress (--no-resume)")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PDFs through a local, model-agnostic OCR pipeline.")
    parser.add_argument("workspace", help="Local filesystem path where work and results are stored.")
    parser.add_argument(
        "--pdfs",
        nargs="*",
        default=None,
        help="Local PDF/image paths, a local glob (e.g. 'docs/*.pdf'), .tar.gz tarballs, or a .txt file listing paths.",
    )

    parser.add_argument("--ocr-model", dest="ocr_model_name", default=DEFAULT_MODEL, choices=sorted(MODEL_REGISTRY), help="OCR model adapter to use.")
    parser.add_argument("--model", default=None, help="Served model id sent to the server / hugging face path for an internal server.")

    parser.add_argument("--pages_per_group", type=int, default=100, help="Aim for this many PDF pages per work item group.")
    parser.add_argument("--max_page_retries", type=int, default=8, help="Max number of times to retry a page.")
    parser.add_argument("--max_page_error_rate", type=float, default=0.004, help="Allowable fraction of fallback pages per document.")
    parser.add_argument("--workers", type=int, default=4, help="Max number of page groups processed at once.")
    parser.add_argument("--max_concurrent_requests", type=int, default=500, help="Max requests in-flight to the inference provider at once.")
    parser.add_argument("--max_server_ready_timeout", type=int, default=600, help="Seconds to wait for the server to become ready.")
    parser.add_argument("--apply_filter", action="store_true", help="Apply basic English/non-spam/non-form PDF filtering.")
    parser.add_argument("--stats", action="store_true", help="Report workspace statistics instead of running any job.")
    parser.add_argument("--markdown", action="store_true", help="Also write per-document Markdown mirroring the input folder structure.")
    parser.add_argument("--target_longest_image_dim", type=int, default=1288, help="Longest-side dimension for rendered page images.")
    parser.add_argument("--guided_decoding", action="store_true", help="Enable guided decoding when the model adapter provides a regex.")

    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", action="store_true", default=True, help="Skip already-completed work items (default).")
    resume_group.add_argument("--no-resume", dest="resume", action="store_false", help="Ignore prior progress and reprocess the workspace from scratch.")

    parser.add_argument(
        "--disk_logging",
        type=str,
        nargs="?",
        const="paperscale-pipeline-debug.log",
        default=None,
        help="Write logs to disk, optionally specify a filename.",
    )

    server_group = parser.add_argument_group("Server arguments")
    server_group.add_argument("--server", type=str, help="URL of an external OpenAI-compatible server (e.g. http://host:port/v1). Skips the internal vLLM server.")
    server_group.add_argument("--api_key", type=str, default=None, help="API key for an authenticated remote server.")

    vllm_group = parser.add_argument_group("VLLM arguments", "Used only for the internal server. Unrecognized args are forwarded to vLLM.")
    vllm_group.add_argument("--gpu-memory-utilization", type=float, help="Fraction of VRAM vLLM may pre-allocate for KV-cache.")
    vllm_group.add_argument("--max_model_len", type=int, default=16384, help="Upper bound (tokens) for vLLM KV-cache allocation.")
    vllm_group.add_argument("--tensor-parallel-size", "-tp", type=int, default=1, help="Tensor parallel size for vLLM.")
    vllm_group.add_argument("--data-parallel-size", "-dp", type=int, default=1, help="Data parallel size for vLLM.")
    vllm_group.add_argument("--port", type=int, default=30024, help="Port for the internal vLLM server.")

    return parser


async def main():
    parser = _build_arg_parser()
    args, unknown_args = parser.parse_known_args()

    if args.disk_logging:
        file_handler = logging.FileHandler(args.disk_logging, mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)
        server_logger.addHandler(file_handler)

    if args.workspace.startswith("s3://"):
        raise SystemExit("error: paperscale only supports local workspaces (S3 support was removed).")

    # Resolve the OCR model adapter and the served model name.
    args.ocr_model = build_ocr_model(args.ocr_model_name)
    if args.model is None:
        args.model = args.ocr_model.default_model_name

    use_internal_server = not args.server

    global max_concurrent_requests_limit
    max_concurrent_requests_limit = asyncio.BoundedSemaphore(args.max_concurrent_requests)

    # We need poppler to render/inspect pdfs.
    check_poppler_version()

    work_queue = WorkQueue(LocalBackend(args.workspace))

    if not args.resume:
        _wipe_workspace_progress(args.workspace)

    if args.pdfs:
        logger.info("Got --pdfs argument, going to add to the work queue")
        pdf_work_paths, tarball_paths = _expand_pdf_inputs(args.pdfs)
        logger.info(f"Found {len(pdf_work_paths):,} pdf/image paths and {len(tarball_paths):,} tarballs to add")

        if pdf_work_paths:
            items_per_group = _estimate_items_per_group(pdf_work_paths, args.pages_per_group)
            await work_queue.populate_queue(list(pdf_work_paths), items_per_group)
        if tarball_paths:
            await work_queue.populate_queue(list(tarball_paths), 1)

    if args.stats:
        print_stats(args)
        return

    # From here on we do inference and (for an internal server) need a GPU.
    if use_internal_server:
        check_torch_gpu_available()

    logger.info(f"Starting pipeline with PID {os.getpid()}")

    if use_internal_server:
        model_name_or_path = await download_model(args.model)
        args.server = f"http://localhost:{args.port}/v1"
        logger.info(f"Using internal server at {args.server}")
    else:
        logger.info(f"Using external server at {args.server}")
        model_name_or_path = None

    qsize = await work_queue.initialize_queue()
    if qsize == 0:
        logger.info("No work to do, exiting")
        return

    vllm_server = None
    if use_internal_server:
        vllm_server = asyncio.create_task(vllm_server_host(model_name_or_path, args, unknown_args))

    await vllm_server_ready(args)

    metrics_task = asyncio.create_task(metrics_reporter(work_queue))

    worker_tasks = [asyncio.create_task(worker(args, work_queue, worker_id=i)) for i in range(args.workers)]
    await asyncio.gather(*worker_tasks)

    if vllm_server is not None:
        vllm_server.cancel()
    metrics_task.cancel()

    tasks_to_wait: list[asyncio.Task] = [metrics_task]
    if vllm_server is not None:
        tasks_to_wait.append(vllm_server)
    await asyncio.gather(*tasks_to_wait, return_exceptions=True)

    _log_final_metrics(args)
    logger.info("Work done")


def _log_final_metrics(args) -> None:
    metrics_summary = metrics.get_metrics_summary()
    total_metrics = metrics_summary["total_metrics"]
    rates = metrics_summary["rates"]

    logger.info("=" * 80)
    logger.info("FINAL METRICS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total elapsed time: {metrics_summary['elapsed_time_seconds']:.2f} seconds")
    logger.info(f"Total Server Input tokens: {total_metrics.get('server_input_tokens', 0):,}")
    logger.info(f"Total Server Output tokens: {total_metrics.get('server_output_tokens', 0):,}")
    logger.info(f"Completed pages: {total_metrics.get('completed_pages', 0):,}")
    logger.info(f"Failed pages: {total_metrics.get('failed_pages', 0):,}")

    completed = total_metrics.get("completed_pages", 0)
    failed = total_metrics.get("failed_pages", 0)
    logger.info(f"Page Failure rate: {failed / max(completed + failed, 1) * 100:.2f}%")

    if "server_output_tokens_per_sec" in rates:
        logger.info(f"Server Output tokens/sec rate: {rates['server_output_tokens_per_sec']:.2f}")


def cli_main():
    """Synchronous entry point for the CLI."""
    return asyncio.run(main())


if __name__ == "__main__":
    cli_main()
