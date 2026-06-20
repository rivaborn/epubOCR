"""Layout-structured blocks from Surya's LayoutPredictor + recognized text lines.

Turns a page into ordered semantic blocks — headings, paragraphs, lists, footnotes,
tables, code — in reading order, and drops running headers/footers/page numbers. This
is the structure layer Marker provides, built natively on Surya's own predictors (so it
stays Apache-code + keeps our per-page confidence and facsimile routing). Opt-in via the
Surya engine's ``layout=True``; the linear OCR text and confidence are unchanged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_TAG = re.compile(r"</?[a-zA-Z][^>]*>")  # Surya emits inline <b>/<i>/<math> markup


def _plain(text: str) -> str:
    return _TAG.sub("", text).strip()

# Surya layout labels (values of surya.layout.LAYOUT_PRED_RELABEL).
_HEADING = {"SectionHeader", "Title"}
_DROP = {"PageHeader", "PageFooter"}        # running heads / page numbers
_LIST = {"ListItem"}
_FOOTNOTE = {"Footnote"}
_CODE = {"Code", "Equation"}
_TABLE = {"Table"}
_FIGURE = {"Picture", "Figure"}
# everything else (Text, Caption, TableOfContents, Form) -> paragraph


@dataclass
class Block:
    type: str               # heading | paragraph | list | footnote | code | table | figure
    text: str = ""
    level: int = 2
    html: str | None = None  # prebuilt markup (tables)

    def to_json(self) -> dict:
        d = {"type": self.type, "text": self.text}
        if self.type == "heading":
            d["level"] = self.level
        if self.html:
            d["html"] = self.html
        return d


def poly_bbox(polygon) -> tuple[float, float, float, float]:
    xs = [float(p[0]) for p in polygon]
    ys = [float(p[1]) for p in polygon]
    return (min(xs), min(ys), max(xs), max(ys))


def _center(bbox) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _contains(region, pt) -> bool:
    return region[0] <= pt[0] <= region[2] and region[1] <= pt[1] <= region[3]


def _block_for(label: str, text: str) -> Block | None:
    if not text:
        return None
    if label in _HEADING:
        return Block("heading", text, level=2)
    if label in _LIST:
        return Block("list", text)
    if label in _FOOTNOTE:
        return Block("footnote", text)
    if label in _CODE:
        return Block("code", text)
    if label in _TABLE:
        return Block("table", text)       # html filled in later if table-rec runs
    return Block("paragraph", text)


def build_blocks(lines: list[tuple[str, tuple]], layout_boxes) -> list[Block]:
    """Build ordered semantic blocks from recognized ``lines`` (text, bbox) + layout regions.

    Walks lines in **recognition reading order** (robust to a noisy layout model) and uses
    layout only to *label* each line (by which region contains its center) and to drop
    running headers/footers. Consecutive same-label lines merge into one block; markup that
    Surya emits inline (``<b>``/``<i>``/…) is stripped.
    """
    region_bboxes = [(poly_bbox(b.polygon), b.label) for b in layout_boxes]

    def label_for(bbox) -> str:
        pt = _center(bbox)
        for rb, lab in region_bboxes:
            if _contains(rb, pt):
                return lab
        return "Text"

    blocks: list[Block] = []
    cur_label: str | None = None
    cur_lines: list[str] = []

    def flush():
        nonlocal cur_lines, cur_label
        text = _plain(" ".join(cur_lines))
        block = _block_for(cur_label or "Text", text)
        if block:
            blocks.append(block)
        cur_lines = []

    for text, bbox in lines:
        if not text.strip():
            continue
        label = label_for(bbox)
        if label in _DROP:                        # running heads/footers only — safe to drop
            flush()
            cur_label = None
            continue
        if label in _FIGURE:                      # keep the text (a mislabeled text block) as prose
            label = "Text"
        # New block on a label change (headers separate from body, lists from paragraphs).
        if label != cur_label:
            flush()
            cur_label = label
        cur_lines.append(text)
    flush()
    return blocks
