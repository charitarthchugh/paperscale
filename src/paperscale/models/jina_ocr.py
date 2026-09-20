"""Jina-OCR-v1 adapter: DeepSeek-OCR-derived MoE parser, full-page Markdown.

jina-ocr-v1 (https://huggingface.co/jinaai/jina-ocr-v1) is a 3.4B document
parser from Jina AI built on the DeepSeek-OCR backbone: a DeepEncoder vision
tower that maps a 1024x1024 global view to 256 visual tokens plus dynamic local
tiles, and a mixture-of-experts decoder with only ~570M active parameters per
token. One page image plus the vendor instruction yields the whole page as
Markdown, so parsing reuses the generic Markdown path. It scores 91.14 on
OmniDocBench v1.6 and 83.4 on olmOCR-Bench at 2.57 pages/s (A100, concurrency
32) -- the shortest outputs (~1085 tokens/page) of any system above 83.
Selected with ``--ocr-model jina-ocr``.

**Licensed CC BY-NC 4.0** -- non-commercial only, unlike the other adapters
here. Commercial use needs a license from Jina AI.

Prompt: the vendor's ``DEFAULT_OCR_PROMPT``, copied verbatim from
``deepseek_ocr_mtp.py`` in the snapshot. The model card also documents a
stricter benchmark instruction (the one the scores above were measured with)
that forces ``$``-delimited LaTeX and HTML tables, drops headers, footers and
figures, and answers ``null`` on a blank page. paperscale sends the
*recommended default* instead: it keeps running headers and footers, which
carry page numbers and case citations in scanned legal documents, and it needs
no ``null`` sentinel handling in :meth:`parse`.

Output shape: plain GitHub-flavored Markdown with Markdown tables. No grounding
tokens (unlike ``unlimited-ocr``), no bbox placeholders (unlike ``ovisocr2``)
and no think block, so :class:`~paperscale.models.markdown.MarkdownModel`
parses it as-is.

Resolution: ``processor_config.json`` pins ``base_size=1024``,
``image_size=640``, ``crop_mode=true`` -- DeepSeek-OCR's "gundam" tiling, one
1024px global view plus 640px local tiles. Pages therefore render at 1024px,
the same as ``unlimited-ocr``; raise it with ``--target_longest_image_dim``
only if fine print needs more local tiles.

Serving. The hosted endpoint is OpenAI-compatible and needs no local GPU::

    paperscale ./workspace --pdfs 'docs/*.pdf' --ocr-model jina-ocr \
        --server https://api.jina.ai/v1 --api_key "$JINA_API_KEY" \
        --model jina-ocr-v1

A cold start there answers HTTP 503; the pipeline's retry/backoff covers it.

Self-hosting needs more than ``vllm serve jinaai/jina-ocr-v1``. The checkpoint
carries FastMTP speculative-decoding tensors (``mtp_module.*``,
``mtp_embed_tokens.*``) that stock vLLM's ``DeepseekOCRForCausalLM`` loader
rejects; the vendor's ``deepseek_ocr_mtp.register()`` installs a subclass that
drops them ("so AutoWeightsLoader does not reject them") and maps the MTP
architectures, including the ``EagleDeepSeekMTPModel`` name vLLM resolves when
``method="eagle"``. The vendor documents that call only for the offline
``LLM(...)`` path, but it has to run in *every* vLLM process, so expose it as a
``vllm.general_plugins`` entry point -- vLLM loads that group in the API server,
the engine core and the workers alike::

    # jina_ocr_plugin.py, in a tiny package installed next to vLLM
    def register() -> None:
        import sys

        from huggingface_hub import snapshot_download

        sys.path.insert(0, snapshot_download("jinaai/jina-ocr-v1"))
        import deepseek_ocr_mtp

        deepseek_ocr_mtp.register()

    # pyproject.toml
    # [project.entry-points."vllm.general_plugins"]
    # jina_ocr = "jina_ocr_plugin:register"

With that installed, the vendor's ``vllm_llm_kwargs()`` becomes serve flags::

    vllm serve jinaai/jina-ocr-v1 --port 8000 --trust-remote-code --dtype bfloat16 \
        --hf-overrides '{"architectures":["DeepseekOCRForCausalLMOCR"],"num_nextn_predict_layers":1,"mtp_recursive":true,"image_token_index":128815}' \
        --speculative-config '{"model":"jinaai/jina-ocr-v1","num_speculative_tokens":3,"method":"eagle"}'

The ``hf-overrides`` are required with or without speculation; drop
``--speculative-config`` to serve the 3B MoE decoder alone. ``method`` must be
``"eagle"`` and not ``"mtp"``: FastMTP feeds each draft step the previous step's
hidden state, while vLLM's ``"mtp"`` re-grounds every step on the target, which
collapses the acceptance rate. It cannot corrupt output either way -- greedy
verification accepts only a token-equality prefix. ``num_speculative_tokens`` is
the vendor's K=3, and speculation pays off at low concurrency and fades as the
batch fills, so read the engine's ``SpecDecoding metrics`` acceptance line
before assuming it speeds up a run with many ``--workers``.

None of that reaches this adapter: paperscale only ever speaks OpenAI HTTP to
whatever is listening.
"""

from __future__ import annotations

from paperscale.models.markdown import MarkdownModel

# Vendor inference prompt, verbatim from DEFAULT_OCR_PROMPT in the snapshot's
# deepseek_ocr_mtp.py (and the "Recommended default" block on the model card).
JINA_OCR_PROMPT = "Transcribe the provided document image into a clean Markdown format, preserving the natural reading order."


class JinaOCRModel(MarkdownModel):
    """Drives jina-ocr-v1, which transcribes a page image to full-page Markdown."""

    default_model_name = "jinaai/jina-ocr-v1"
    preferred_longest_image_dim = 1024

    def __init__(self) -> None:
        super().__init__(prompt=JINA_OCR_PROMPT)

    def build_messages(self, image_base64: str) -> list[dict]:
        # Image part first, then the instruction: that is the order the vendor's
        # own inference code uses (prepare_vllm_input, example.py), and the chat
        # template emits a newline after a trailing image part, reproducing the
        # "<image>\n<instruction>" sequence the model was trained on. The card's
        # hosted-API curl shows text first; the vendor's script wins.
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    {"type": "text", "text": self._prompt},
                ],
            }
        ]

    def sampling_params(self) -> dict:
        """Extra OpenAI sampling params for jina-ocr-v1.

        TODO(decide): how much of the vendor's anti-loop protection to send.

        The vendor guards against decode loops twice, neither of which travels
        over plain OpenAI HTTP for free:

        * ``vllm_sampling_params()`` sets ``repetition_penalty=1.05`` and a
          ``repetition_detection`` n-gram stop (``max_pattern_size=35``,
          ``min_pattern_size=35``, ``min_count=10``).
        * the Transformers path attaches ``SlidingWindowNoRepeatNgramProcessor``
          (``no_repeat_ngram_size=35``, ``ngram_window=1024``), whitelisting the
          ``<td>``/``</td>`` tokens so real tables may repeat.

        Temperature stays pipeline-owned (it escalates per retry attempt).
        """
        return {}
