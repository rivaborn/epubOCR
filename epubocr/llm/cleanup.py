"""LLM cleanup passes (epubOCR.md §7).

Each pass turns OCR ground truth into clean XHTML *without inventing content*. The
per-page pass is wired here as the reference; chapter-assembly / structure /
proofreading passes follow the same shape (client + prompt + fidelity gate).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import Config
from .client import LLMClient
from .fidelity_verifier import Verdict, verify

PAGE_CLEANUP_PROMPT = (
    "You are cleaning OCR from a scanned book page. The OCR text is authoritative.\n"
    "- Do NOT add facts, words, or sentences that are not in the OCR text.\n"
    "- Preserve original wording; repair an OCR error only when unambiguous.\n"
    "- Rejoin broken lines into paragraphs; drop running headers/footers/page numbers.\n"
    "- Preserve headings, italics, footnotes, lists, and tables.\n"
    "- Return ONLY a valid XHTML fragment (no <html>/<body> wrapper, no commentary)."
)

PROMPT_VERSION = "page-cleanup-1"


@dataclass
class CleanupResult:
    xhtml: str           # the trusted output (LLM result, or deterministic fallback)
    verdict: Verdict
    used_fallback: bool


def clean_page(config: Config, ocr_text: str, deterministic_fallback: str, *,
               endpoint_name: str | None = None, model: str | None = None) -> CleanupResult:
    """Run the per-page cleanup pass behind the fidelity verifier.

    On a fidelity failure, return ``deterministic_fallback`` (the OCR text after
    the deterministic layout pass) and flag it, rather than trusting invented text.
    ``endpoint_name`` / ``model`` override the configured text-cleanup role (e.g. to
    reuse an already-loaded model and avoid a second GPU model swap).
    """
    client = LLMClient(config.endpoint(endpoint_name)) if endpoint_name else \
        LLMClient.for_role(config, "text_cleanup")
    model = model or config.model("text_xhtml", endpoint=client.endpoint.name)
    messages = [
        {"role": "system", "content": PAGE_CLEANUP_PROMPT},
        {"role": "user", "content": ocr_text},
    ]
    xhtml = client.chat(model, messages, temperature=0.0)
    verdict = verify(ocr_text, xhtml, config.fidelity)
    if verdict.ok:
        return CleanupResult(xhtml=xhtml, verdict=verdict, used_fallback=False)
    return CleanupResult(xhtml=deterministic_fallback, verdict=verdict, used_fallback=True)
