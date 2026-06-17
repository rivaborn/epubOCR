"""Fidelity / structure metrics for the eval harness and the fidelity verifier.

CER / WER measure transcription fidelity; ``insertion_rate`` approximates the
hallucination risk that the fidelity-first design exists to suppress.
"""
from __future__ import annotations

import re
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
