"""Generate a tiny PDF fixture: one born-digital text page + one image-only (scanned) page.

Exercises PDF page classification — the text-layer page must be preserved as TEXT, the
image page must be classified IMAGE and OCR'd. Authored with PyMuPDF (no extra deps).
Run: ``uv run python tests/fixtures/make_sample_pdf.py`` -> writes ``tests/fixtures/sample.pdf``.
"""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
OUT = HERE / "sample.pdf"


def _scanned_png(text: str, size=(1000, 1400)) -> bytes:
    img = Image.new("L", size, color=255)
    d = ImageDraw.Draw(img)
    for i, line in enumerate(text.splitlines()):
        d.text((80, 100 + i * 44), line, fill=20)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def build() -> Path:
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz

    doc = fitz.open()
    # Page 1: a real text layer (born-digital) -> classifier marks TEXT, pipeline preserves it.
    p1 = doc.new_page(width=595, height=842)  # A4 in points
    p1.insert_text((72, 100), "Chapter 1", fontsize=20)
    p1.insert_text((72, 150),
                   "This page has a real text layer, so the classifier should mark it TEXT\n"
                   "and the pipeline should preserve it rather than OCR it.", fontsize=12)
    # Page 2: a full-page image with no text layer -> IMAGE, should be OCR'd downstream.
    p2 = doc.new_page(width=595, height=842)
    png = _scanned_png("Page two is a scanned image.\nThe quick brown fox\njumps over the lazy dog.")
    p2.insert_image(p2.rect, stream=png)

    doc.save(str(OUT))
    doc.close()
    return OUT


if __name__ == "__main__":
    print("wrote", build())
