"""Fidelity / structure metrics for the eval harness and the fidelity verifier.

CER / WER measure transcription fidelity; ``insertion_rate`` approximates the
hallucination risk that the fidelity-first design exists to suppress.
"""
from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein

_WORD = re.compile(r"[^\W_]+", re.UNICODE)  # alphanumeric runs, no underscores/punct


def _norm(s: str) -> str:
    return " ".join(s.split())


def cer(ref: str, hyp: str) -> float:
    """Character error rate = edit distance / len(ref). 0.0 == identical."""
    ref, hyp = _norm(ref), _norm(hyp)
    if not ref:
        return 0.0 if not hyp else 1.0
    return Levenshtein.distance(ref, hyp) / len(ref)


def wer(ref: str, hyp: str) -> float:
    """Word error rate = token edit distance / token count of ref."""
    r, h = ref.split(), hyp.split()
    if not r:
        return 0.0 if not h else 1.0
    return Levenshtein.distance(r, h) / len(r)


def tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD.finditer(text)]


def insertion_rate(ocr_text: str, candidate_text: str) -> float:
    """Fraction of candidate word-tokens absent (as a multiset) from the OCR text.

    A proxy for hallucination: words the cleanup introduced that the OCR — our
    ground truth — never saw. 0.0 means the candidate only reordered/dropped.
    """
    cand = tokens(candidate_text)
    if not cand:
        return 0.0
    available = Counter(tokens(ocr_text))
    inserted = 0
    for t in cand:
        if available.get(t, 0) > 0:
            available[t] -= 1
        else:
            inserted += 1
    return inserted / len(cand)


def net_length_delta(ref: str, hyp: str) -> float:
    """|len(hyp) - len(ref)| / len(ref) over normalized text."""
    ref, hyp = _norm(ref), _norm(hyp)
    if not ref:
        return 0.0 if not hyp else 1.0
    return abs(len(hyp) - len(ref)) / len(ref)


def repetition_ratio(text: str) -> float:
    """Share of the most frequent word-token — a degeneracy signal.

    A healthy page is well below 0.1; a VLM stuck in a '[illegible] [illegible]…'
    loop approaches 1.0. Used to flag OCR output the fidelity verifier can't (it
    only guards the cleanup pass, not the OCR pass itself).
    """
    toks = tokens(text)
    if len(toks) < 20:
        return 0.0
    return Counter(toks).most_common(1)[0][1] / len(toks)


def is_degenerate(text: str, *, max_repetition: float = 0.35) -> bool:
    return repetition_ratio(text) > max_repetition


def length_outliers(lengths: dict[int, int], *, factor: float = 2.0,
                    min_pages: int = 12, min_chars: int = 1200) -> set[int]:
    """Pages whose OCR text is far longer than the book's own median page.

    The second degeneracy test, and the one that catches what ``repetition_ratio``
    cannot. That test keys on the single most frequent *token*, so a model looping over
    *varied* filler slips under it: measured 2026-08-06, PaddleOCR-VL ran 12 of 359 pages
    to the token cap emitting "Input the name of the hospital." / "The text is not
    printed." at repetition ratios of 0.03-0.33 — all but one below the 0.35 threshold —
    and Chandra fabricated 12,147 chars on a *blank* page at ratio 0.101. Every one of
    those pages was 2x+ the book's median length, so this catches all 13 for free.

    Scale-free by construction: the reference is the book's own median, so a densely-set
    book is not penalised for being dense. ``min_chars`` keeps a book of near-empty pages
    (a plate section) from making every real page an "outlier"; ``min_pages`` declines to
    guess a median from too small a sample.
    """
    real = [n for n in lengths.values() if n > 0]
    if len(real) < min_pages:
        return set()
    med = statistics.median(real)
    threshold = max(factor * med, min_chars)
    return {idx for idx, n in lengths.items() if n > threshold}


@dataclass(frozen=True)
class PageScore:
    engine: str
    cer: float
    wer: float
    insertion_rate: float

    def as_row(self) -> dict:
        return {
            "engine": self.engine,
            "cer": round(self.cer, 4),
            "wer": round(self.wer, 4),
            "insertion_rate": round(self.insertion_rate, 4),
        }


def score_engine(engine: str, gold_text: str, ocr_text: str) -> PageScore:
    return PageScore(
        engine=engine,
        cer=cer(gold_text, ocr_text),
        wer=wer(gold_text, ocr_text),
        insertion_rate=insertion_rate(gold_text, ocr_text),
    )
