"""Fidelity verifier (epubOCR.md §7).

After an LLM cleanup pass, compare the cleaned XHTML against the OCR ground truth.
If the cleaned text drifts past the configured thresholds — too much character
change, too many inserted words, or too big a length swing — the page is *held*:
the caller falls back to the deterministically-cleaned OCR and flags it for QA.

This operationalizes the prompt rule "do not invent content." The prompt asks;
this checks.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field

from ..config import FidelityThresholds
from ..eval import metrics

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def xhtml_to_text(fragment: str) -> str:
    """Strip tags + unescape entities so we compare prose, not markup."""
    return _WS.sub(" ", html.unescape(_TAG.sub(" ", fragment))).strip()


@dataclass(frozen=True)
class Verdict:
    ok: bool
    cer: float
    insertion_rate: float
    net_length_delta: float
    reasons: list[str] = field(default_factory=list)


def verify(ocr_text: str, cleaned_xhtml: str, thresholds: FidelityThresholds) -> Verdict:
    cleaned_text = xhtml_to_text(cleaned_xhtml)
    c = metrics.cer(ocr_text, cleaned_text)
    ins = metrics.insertion_rate(ocr_text, cleaned_text)
    dl = metrics.net_length_delta(ocr_text, cleaned_text)

    reasons: list[str] = []
    if c > thresholds.max_cer_vs_ocr:
        reasons.append(f"CER {c:.3f} > {thresholds.max_cer_vs_ocr}")
    if ins > thresholds.max_insertion_ratio:
        reasons.append(f"insertion {ins:.3f} > {thresholds.max_insertion_ratio}")
    if dl > thresholds.max_net_length_delta:
        reasons.append(f"length delta {dl:.3f} > {thresholds.max_net_length_delta}")

    return Verdict(ok=not reasons, cer=c, insertion_rate=ins, net_length_delta=dl, reasons=reasons)
