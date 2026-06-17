"""EPUB reconstruction with per-page adaptive output (epubOCR.md §8).

Reflowable XHTML for prose; facsimile fallback (page image + hidden OCR text layer)
for hard/low-confidence pages. Builds EPUB3 with EbookLib and preserves print
pagination via inline pagebreak anchors + a page-list nav.
"""
from __future__ import annotations

import html
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class OutputMode(str, Enum):
    REFLOWABLE = "reflowable"
    FACSIMILE = "facsimile"


@dataclass
class SpineDoc:
    index: int
    title: str
    mode: OutputMode
    page_number: int | None = None
    body_xhtml: str | None = None     # inner body for reflowable pages
    image_path: Path | None = None    # source image for facsimile pages
    ocr_text: str | None = None       # hidden, searchable text layer for facsimile


def choose_mode(page_type: str, mean_conf: float | None, *, conf_floor: float = 0.80,
                fidelity_held: bool = False) -> OutputMode:
    """Reflowable vs facsimile for one page (epubOCR.md §8)."""
    if page_type in ("cover", "image_only_art") or fidelity_held:
        return OutputMode.FACSIMILE
    if mean_conf is not None and mean_conf < conf_floor:
        return OutputMode.FACSIMILE
    return OutputMode.REFLOWABLE


def pagebreak_anchor(page_number: int) -> str:
    return (f'<span epub:type="pagebreak" role="doc-pagebreak" '
            f'id="page-{page_number}" aria-label="{page_number}"></span>')


def paragraphs_to_xhtml(text: str) -> str:
    """Wrap cleaned plain text (blank-line separated) into <p> elements."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    return "\n".join(f"<p>{html.escape(p)}</p>" for p in paras) or "<p></p>"


_XHTML = (
    '<?xml version="1.0" encoding="utf-8"?>\n'
    '<!DOCTYPE html>\n'
    '<html xmlns="http://www.w3.org/1999/xhtml" '
    'xmlns:epub="http://www.idpf.org/2007/ops" lang="{lang}">'
    "<head><title>{title}</title></head><body>{body}</body></html>"
)


def _doc_xhtml(doc: SpineDoc, lang: str, image_filename: str | None) -> str:
    anchor = pagebreak_anchor(doc.page_number) if doc.page_number is not None else ""
    if doc.mode is OutputMode.FACSIMILE and image_filename:
        hidden = ""
        if doc.ocr_text:
            hidden = (f'<div aria-hidden="false" style="position:absolute;left:-9999px;">'
                      f'{html.escape(doc.ocr_text)}</div>')
        body = (f'{anchor}<section epub:type="page">'
                f'<img src="images/{image_filename}" alt="{html.escape(doc.title)}" '
                f'style="max-width:100%;"/>{hidden}</section>')
    else:
        body = f'{anchor}<section>{doc.body_xhtml or "<p></p>"}</section>'
    return _XHTML.format(lang=lang, title=html.escape(doc.title), body=body)


def _nav_xhtml(lang: str, toc: list[tuple[str, str]], pages: list[tuple[int, str]]) -> str:
    toc_items = "".join(f'<li><a href="{href}">{html.escape(t)}</a></li>' for t, href in toc)
    page_items = "".join(f'<li><a href="{href}">{n}</a></li>' for n, href in pages)
    page_nav = (f'<nav epub:type="page-list" hidden=""><ol>{page_items}</ol></nav>'
                if pages else "")
    body = (f'<nav epub:type="toc" id="toc"><h1>Contents</h1><ol>{toc_items}</ol></nav>'
            f'{page_nav}')
    return _XHTML.format(lang=lang, title="Navigation", body=body)


def build_epub(docs: list[SpineDoc], *, title: str, output_path: Path,
               language: str = "en", identifier: str = "urn:uuid:epubocr") -> Path:
    from ebooklib import epub  # lazy

    book = epub.EpubBook()
    book.set_identifier(identifier)
    book.set_title(title)
    book.set_language(language)

    spine_items = []
    links: list = []
    toc: list[tuple[str, str]] = []
    page_list: list[tuple[int, str]] = []
    seen_images: set[str] = set()

    for doc in docs:
        fname = f"p{doc.index:04d}.xhtml"
        image_filename = None
        if doc.mode is OutputMode.FACSIMILE and doc.image_path and doc.image_path.exists():
            image_filename = doc.image_path.name
            if image_filename not in seen_images:
                book.add_item(epub.EpubImage(
                    uid=f"img-{doc.index}", file_name=f"images/{image_filename}",
                    media_type=_media_type(image_filename), content=doc.image_path.read_bytes()))
                seen_images.add(image_filename)

        # Raw EpubItem (not EpubHtml) so our exact XHTML — pagebreak anchors, facsimile
        # <img>, epub: namespace — is preserved verbatim instead of ebooklib re-templating it.
        item = epub.EpubItem(uid=f"p{doc.index}", file_name=fname,
                             media_type="application/xhtml+xml",
                             content=_doc_xhtml(doc, language, image_filename).encode("utf-8"))
        book.add_item(item)
        spine_items.append(item)
        links.append(epub.Link(fname, doc.title, f"p{doc.index}"))
        toc.append((doc.title, fname))
        if doc.page_number is not None:
            page_list.append((doc.page_number, f"{fname}#page-{doc.page_number}"))

    # Custom nav as a raw EpubItem so we control the page-list verbatim — EpubNav
    # auto-gen omits it, and an 'nav'-flagged EpubHtml gets templated away to empty.
    nav = epub.EpubItem(uid="nav", file_name="nav.xhtml", media_type="application/xhtml+xml",
                        content=_nav_xhtml(language, toc, page_list).encode("utf-8"))
    nav.properties = ["nav"]
    book.add_item(nav)
    book.add_item(epub.EpubNcx())

    book.toc = tuple(links)
    book.spine = [nav, *spine_items]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    epub.write_epub(str(output_path), book)
    return output_path


def _media_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".gif": "image/gif", ".svg": "image/svg+xml"}.get(ext, "image/png")
