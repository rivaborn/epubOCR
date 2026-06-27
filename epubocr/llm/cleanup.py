"""LLM cleanup passes (epubOCR.md §7).

Each pass turns OCR ground truth into clean XHTML *without inventing content*. The
per-page pass is wired here as the reference; chapter-assembly / structure /
proofreading passes follow the same shape (client + prompt + fidelity gate).
"""
from __future__ import annotations

from dataclasses import dataclass

from lxml import etree

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


def _wellformed_xhtml(fragment: str) -> str | None:
    """Re-serialize an LLM XHTML fragment as well-formed XML, recovering from minor markup
    errors (a stray ``&``, an unclosed tag, ``<br>`` for ``<br/>``).

    ``assemble`` writes page XHTML verbatim (raw ``EpubItem``), and the fidelity verifier
    only compares stripped *text* — so malformed-but-faithful LLM output would otherwise
    produce an invalid EPUB. Wrap in a namespaced root (so ``epub:type`` resolves), parse
    with recovery, then return the inner markup. ``None`` when nothing parseable remains, so
    the caller falls back to the deterministic cleanup.
    """
    wrapped = f'<f xmlns:epub="http://www.idpf.org/2007/ops">{fragment}</f>'
    root = etree.fromstring(wrapped, parser=etree.XMLParser(recover=True))
    if root is None:
        return None
    full = etree.tostring(root, encoding="unicode")
    close = full.rfind("</f>")
    if close == -1:                       # recovered to an empty / self-closed wrapper
        return None
    return full[full.index(">") + 1:close].strip() or None


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
    xhtml = _wellformed_xhtml(client.chat(model, messages, temperature=0.0))
    if xhtml is None:                     # unparseable markup -> trust deterministic cleanup
        return CleanupResult(
            xhtml=deterministic_fallback,
            verdict=Verdict(ok=False, cer=1.0, insertion_rate=0.0, net_length_delta=1.0,
                            reasons=["LLM returned unparseable XHTML"]),
            used_fallback=True)
    verdict = verify(ocr_text, xhtml, config.fidelity)
    if verdict.ok:
        return CleanupResult(xhtml=xhtml, verdict=verdict, used_fallback=False)
    return CleanupResult(xhtml=deterministic_fallback, verdict=verdict, used_fallback=True)
