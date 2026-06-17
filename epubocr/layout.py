"""Deterministic layout cleanup (epubOCR.md §6).

These run *before* the LLM and handle the mechanical work cheaply and reliably:
de-hyphenation, running header/footer removal, and within-page paragraph rejoin.
The LLM is reserved for semantic structure (chapters, footnotes, ambiguous joins).
"""
from __future__ import annotations

import re
from collections import Counter

_HYPHEN_WRAP = re.compile(r"(\w+)-\n(\w+)")
_SENT_END = re.compile(r"[.!?\"'’”)\]]\s*$")


def dehyphenate(text: str) -> str:
    """Join words split across a line break by a trailing hyphen: 'exam-\\nple' -> 'example'.

    Conservative: only joins when both sides are word characters. (A dictionary check
    via the optional ``dehyphen`` package can be layered on later for real hyphens.)
    """
    return _HYPHEN_WRAP.sub(lambda m: m.group(1) + m.group(2), text)


def rejoin_paragraphs(text: str) -> str:
    """Merge soft-wrapped lines into paragraphs; keep blank-line breaks as boundaries."""
    out_paras: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        merged = lines[0]
        for ln in lines[1:]:
            if _SENT_END.search(merged):
                merged += "\n" + ln
            else:
                merged += " " + ln
        out_paras.append(merged)
    return "\n\n".join(out_paras)


def find_running_lines(pages: list[str], *, min_fraction: float = 0.5) -> set[str]:
    """Lines (e.g. running heads, page numbers) repeated across >= min_fraction of pages.

    Looks at the first and last non-empty line of each page; page numbers are
    normalized to '#' so '12'/'13' count as the same recurring artifact.
    """
    if not pages:
        return set()
    counts: Counter[str] = Counter()
    for page in pages:
        lines = [ln.strip() for ln in page.splitlines() if ln.strip()]
        if not lines:
            continue
        for cand in {lines[0], lines[-1]}:
            counts[re.sub(r"\d+", "#", cand)] += 1
    threshold = max(2, int(len(pages) * min_fraction))
    return {norm for norm, c in counts.items() if c >= threshold}


def strip_running_lines(page: str, running: set[str]) -> str:
    out = []
    for ln in page.splitlines():
        if re.sub(r"\d+", "#", ln.strip()) in running:
            continue
        out.append(ln)
    return "\n".join(out)


def clean_page_text(text: str) -> str:
    """Full deterministic pass for a single page (no cross-page context)."""
    return rejoin_paragraphs(dehyphenate(text))
