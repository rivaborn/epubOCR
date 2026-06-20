"""Cross-engine consensus (epubOCR.md §4) — a model-agnostic fidelity signal.

Where two independent OCR engines (e.g. Surya + a VLM) agree on a page's text, it is
almost certainly correct — two different architectures converging is strong evidence.
Where they disagree, trust drops, so the page is routed to facsimile / QA. This also
catches VLM confabulation directly: invented words won't match the traditional engine.
"""
from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import fuzz

from .eval import metrics


def agreement(text_a: str, text_b: str) -> float:
    """Token-level agreement in [0, 1] between two OCR outputs of the same page."""
    a = " ".join(metrics.tokens(text_a))
    b = " ".join(metrics.tokens(text_b))
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return fuzz.token_sort_ratio(a, b) / 100.0


@dataclass(frozen=True)
class Consensus:
    agreement: float
    trusted: bool
    reason: str


def assess(primary_text: str, primary_conf: float | None, challenger_text: str, *,
           agree_floor: float = 0.85, conf_floor: float = 0.80) -> Consensus:
    """Decide whether to trust a page's transcription for reflowable output.

    Trust if the engines agree (independent confirmation) OR the primary engine is
    confident on its own. Otherwise route to facsimile.
    """
    agr = agreement(primary_text, challenger_text)
    if agr >= agree_floor:
        return Consensus(agr, True, f"engines agree ({agr:.2f})")
    if primary_conf is not None and primary_conf >= conf_floor:
        return Consensus(agr, True, f"primary confident ({primary_conf:.2f})")
    return Consensus(agr, False, f"low agreement ({agr:.2f}) and low/no confidence")
