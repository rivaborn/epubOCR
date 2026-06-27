"""PDF ingestion + page classification — the PDF front-end (parallels ingest.py).

Everything downstream of ingest is **source-agnostic**: it consumes ``extracted/manifest.json``
+ page images and never looks at the original container. So accepting PDFs is just emitting that
same manifest from a PDF instead of an EPUB. Classification mirrors the EPUB path:

  * a page with a real **text layer** (born-digital, incl. scanned-then-OCR'd PDFs) -> ``text``,
    preserved verbatim rather than re-OCR'd (epubOCR.md §2: real text beats OCR);
  * an **image-only** page -> ``image``, OCR'd downstream. Its bitmap is **lossless-extracted**
    when the page is a single full-page scan (best OCR fidelity), else rendered at 300 DPI.

Uses PyMuPDF (``pymupdf`` / ``fitz``). NOTE: PyMuPDF is **AGPL-3.0** — it is an opt-in extra
(``uv sync --extra pdf``); distributing epubocr with it pulls in that copyleft.
"""
from __future__ import annotations

import html
import shutil
from pathlib import Path

from .ingest import PageType, SpinePage
from .storage import BookProject

_PDF_TEXT_MIN = 24       # extractable real chars above which a page is treated as born-digital text
_FULLPAGE_COVER = 0.80   # image-rect / page-area ratio above which one image *is* the page
_RENDER_DPI = 300        # rasterization DPI for pages we can't extract losslessly


def _import_fitz():
    """Lazy-import PyMuPDF under either of its module names, with an actionable error."""
    try:
        import pymupdf  # modern name (>=1.24)
        return pymupdf
    except ImportError:
        try:
            import fitz  # classic name
            return fitz
        except ImportError as exc:
            raise RuntimeError(
                "PDF ingest needs PyMuPDF — install with `uv sync --extra pdf` "
                "(note: PyMuPDF is AGPL-3.0)."
            ) from exc


def _text_len(page) -> int:
    return len(" ".join(page.get_text("text").split()))


def _text_html(page) -> str:
    """Born-digital page text -> paragraph XHTML (lossy on inline styling, keeps structure)."""
    blocks = page.get_text("blocks")  # tuples: (x0, y0, x1, y1, text, block_no, block_type)
    paras: list[str] = []
    for b in sorted(blocks, key=lambda b: (round(b[1], 1), round(b[0], 1))):  # top->bottom, left->right
        if len(b) >= 7 and b[6] != 0:   # block_type 0 == text; skip image blocks
            continue
        txt = " ".join((b[4] or "").split())
        if txt:
            paras.append(f"<p>{html.escape(txt)}</p>")
    return "\n".join(paras) or "<p></p>"


def _extract_or_render(doc, page, out_dir: Path, stem: str) -> str:
    """Write the page bitmap to ``out_dir`` and return its filename.

    Prefer lossless extraction of the original embedded image when the page is a single
    full-page scan (no re-encode = best OCR fidelity); otherwise rasterize at 300 DPI.
    A rotated page is always rendered so the rotation is baked into the pixels.
    """
    page_area = abs(page.rect.width * page.rect.height)
    if not page.rotation and page_area:
        imgs = page.get_images(full=True)
        if len(imgs) == 1:
            xref = imgs[0][0]
            try:
                rects = page.get_image_rects(xref)
            except Exception:  # noqa: BLE001 - older PyMuPDF / odd images -> fall back to render
                rects = []
            covered = max((abs(r.width * r.height) for r in rects), default=0.0)
            if covered / page_area > _FULLPAGE_COVER:
                info = doc.extract_image(xref)
                ext = (info.get("ext") or "png").lower()
                fname = f"{stem}.{ext}"
                (out_dir / fname).write_bytes(info["image"])
                return fname
    pix = page.get_pixmap(dpi=_RENDER_DPI)
    fname = f"{stem}.png"
    pix.save(str(out_dir / fname))
    return fname


def ingest_pdf(pdf_path: Path, project: BookProject, *, force_ocr: bool = False) -> list[SpinePage]:
    """Classify a PDF's pages and write the same ``manifest.json`` the EPUB path produces.

    ``force_ocr`` ignores any embedded text layer and routes **every** page to image OCR — use
    it for scanned PDFs that carry a poor prior OCR layer you'd rather re-transcribe with Surya.
    """
    fitz = _import_fitz()
    project.ensure()
    pdf_path = Path(pdf_path)
    dest = project.source / pdf_path.name
    if not dest.exists():
        shutil.copy2(pdf_path, dest)

    pages: list[SpinePage] = []
    doc = fitz.open(str(pdf_path))
    try:
        for pno in range(doc.page_count):
            page = doc[pno]
            tlen = _text_len(page)
            idref = f"pdfpage-{pno + 1}"
            if tlen >= _PDF_TEXT_MIN and not force_ocr:
                pages.append(SpinePage(pno, idref, "", PageType.TEXT, tlen,
                                       text_html=_text_html(page)))
            else:
                stem = f"page_{pno + 1:04d}"
                fname = _extract_or_render(doc, page, project.pages, stem)
                pages.append(SpinePage(pno, idref, "", PageType.IMAGE, tlen,
                                       extracted_images=[fname]))
    finally:
        doc.close()

    project.write_json(project.manifest_path, {
        "source_file": pdf_path.name,
        "source": "pdf",
        "force_ocr": force_ocr,
        "page_count": len(pages),
        "counts": {t.value: sum(1 for p in pages if p.page_type is t) for t in PageType},
        "pages": [p.to_json() for p in pages],
    })
    return pages
