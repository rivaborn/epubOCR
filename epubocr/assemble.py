"""EPUB reconstruction: per-page adaptive output + continuous, chapter-aware flow (epubOCR.md §8).

Reflowable XHTML for prose; facsimile fallback (page image + hidden OCR text layer) for
hard/low-confidence pages. Readers force a visual break at every *spine document*, so one
document per scanned page reads as a break at every original page. Instead, pages that
belong together are concatenated into one spine document (a :class:`Section`) so the text
flows; a chapter boundary starts a new Section so chapters stay separate and navigable.
Print pagination is preserved via inline pagebreak anchors + a page-list nav, and a
paragraph split across a page boundary is stitched back together (see ``_stitch_pages``).
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
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


@dataclass
class Section:
    """One spine document: a run of pages that flow together — a chapter, the cover/front
    matter, or (when no chapters are detected) the whole book."""
    id: str
    title: str
    docs: list[SpineDoc] = field(default_factory=list)
    in_toc: bool = True


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


def blocks_to_xhtml(blocks: list[dict]) -> str:
    """Render Surya layout blocks (headings/paragraphs/lists/footnotes/tables) to XHTML."""
    out: list[str] = []
    pending_list: list[str] = []

    def flush_list():
        if pending_list:
            out.append("<ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in pending_list) + "</ul>")
            pending_list.clear()

    for b in blocks:
        btype, text = b.get("type", "paragraph"), b.get("text", "")
        if btype == "list":
            pending_list.append(text)
            continue
        flush_list()
        if btype == "heading":
            lvl = min(max(int(b.get("level", 2)), 1), 6)
            out.append(f"<h{lvl}>{html.escape(text)}</h{lvl}>")
        elif btype == "footnote":
            out.append(f'<aside epub:type="footnote"><p>{html.escape(text)}</p></aside>')
        elif btype == "code":
            out.append(f"<pre>{html.escape(text)}</pre>")
        elif btype == "table":
            out.append(b.get("html") or f'<div class="ocr-table"><p>{html.escape(text)}</p></div>')
        elif btype == "figure":
            continue
        else:
            out.append(f"<p>{html.escape(text)}</p>")
    flush_list()
    return "\n".join(out) or "<p></p>"


_XHTML = (
    '<?xml version="1.0" encoding="utf-8"?>\n'
    '<!DOCTYPE html>\n'
    '<html xmlns="http://www.w3.org/1999/xhtml" '
    'xmlns:epub="http://www.idpf.org/2007/ops" lang="{lang}">'
    "<head><title>{title}</title></head><body>{body}</body></html>"
)

# A paragraph split across a page boundary renders as '<p>...tail</p>{anchor}<p>X...'. Stitch
# it back into one <p> (anchor left inline) only when the tail is NOT sentence-final and the
# next paragraph starts lowercase — a clear mid-sentence continuation. Conservative by design:
# it never merges two genuinely separate paragraphs (those end in '.'/'!'/'?'/quote, or the
# continuation is capitalized). Deterministic-cleanup paragraphs only (no inline tags), so the
# tail capture is exact; the LLM path keeps stitching off.
_SENT_END = re.compile(r"[.!?\"'’”)\]]$")
_STITCH = re.compile(r'([^<>]*)</p>\s*(<span epub:type="pagebreak"[^>]*></span>)\s*<p>(.)')


def _stitch_pages(body: str) -> str:
    def repl(m: re.Match) -> str:
        tail, anchor, first = m.group(1), m.group(2), m.group(3)
        if not _SENT_END.search(html.unescape(tail).strip()) and first.islower():
            return f"{tail}{anchor} {first}"          # merge: anchor stays inline mid-paragraph
        return m.group(0)                              # keep the paragraph break
    return _STITCH.sub(repl, body)


def _doc_body(doc: SpineDoc, image_filename: str | None) -> str:
    """One page's inner markup (pagebreak anchor + content), no <html> wrapper.

    Reflowable pages emit their bare paragraphs (no per-page <section>), so consecutive
    pages' paragraphs are adjacent in the stream and ``_stitch_pages`` can rejoin a split
    one. Facsimile pages keep a <section> wrapper (a distinct image block that breaks flow).
    """
    anchor = pagebreak_anchor(doc.page_number) if doc.page_number is not None else ""
    if doc.mode is OutputMode.FACSIMILE and image_filename:
        hidden = ""
        if doc.ocr_text:
            hidden = (f'<div aria-hidden="false" style="position:absolute;left:-9999px;">'
                      f'{html.escape(doc.ocr_text)}</div>')
        return (f'{anchor}<section epub:type="page">'
                f'<img src="images/{image_filename}" alt="{html.escape(doc.title)}" '
                f'style="max-width:100%;"/>{hidden}</section>')
    if doc.mode is OutputMode.FACSIMILE and doc.ocr_text:
        # Facsimile requested but the page image is unavailable — show the OCR text as
        # visible prose rather than emit a silent blank page.
        return f'{anchor}<section>{paragraphs_to_xhtml(doc.ocr_text)}</section>'
    return f'{anchor}{doc.body_xhtml or "<p></p>"}'


def _nav_xhtml(lang: str, toc: list[tuple[str, str]], pages: list[tuple[int, str]]) -> str:
    toc_items = "".join(f'<li><a href="{href}">{html.escape(t)}</a></li>' for t, href in toc)
    page_items = "".join(f'<li><a href="{href}">{n}</a></li>' for n, href in pages)
    page_nav = (f'<nav epub:type="page-list" hidden=""><ol>{page_items}</ol></nav>'
                if pages else "")
    body = (f'<nav epub:type="toc" id="toc"><h1>Contents</h1><ol>{toc_items}</ol></nav>'
            f'{page_nav}')
    return _XHTML.format(lang=lang, title="Navigation", body=body)


def build_epub(sections: list[Section], *, title: str, output_path: Path,
               language: str = "en", identifier: str = "urn:uuid:epubocr",
               stitch: bool = True) -> Path:
    from ebooklib import epub  # lazy

    book = epub.EpubBook()
    book.set_identifier(identifier)
    book.set_title(title)
    book.set_language(language)

    spine_items: list = []
    page_list: list[tuple[int, str]] = []
    seen_images: set[str] = set()

    for sec in sections:
        fname = f"{sec.id}.xhtml"
        bodies: list[str] = []
        for doc in sec.docs:
            image_filename = None
            if doc.mode is OutputMode.FACSIMILE and doc.image_path and doc.image_path.exists():
                image_filename = doc.image_path.name
                if image_filename not in seen_images:
                    book.add_item(epub.EpubImage(
                        uid=f"img-{doc.index}", file_name=f"images/{image_filename}",
                        media_type=_media_type(image_filename), content=doc.image_path.read_bytes()))
                    seen_images.add(image_filename)
            bodies.append(_doc_body(doc, image_filename))
            if doc.page_number is not None:
                page_list.append((doc.page_number, f"{fname}#page-{doc.page_number}"))

        stream = "".join(bodies)
        if stitch:
            stream = _stitch_pages(stream)
        # Raw EpubItem (not EpubHtml) so our exact XHTML — pagebreak anchors, facsimile
        # <img>, epub: namespace — is preserved verbatim instead of ebooklib re-templating it.
        item = epub.EpubItem(uid=sec.id, file_name=fname, media_type="application/xhtml+xml",
                             content=_XHTML.format(lang=language, title=html.escape(sec.title),
                                                   body=stream).encode("utf-8"))
        book.add_item(item)
        spine_items.append(item)

    toc = [(sec.title, f"{sec.id}.xhtml") for sec in sections if sec.in_toc]
    # Custom nav as a raw EpubItem so we control the page-list verbatim — EpubNav
    # auto-gen omits it, and an 'nav'-flagged EpubHtml gets templated away to empty.
    nav = epub.EpubItem(uid="nav", file_name="nav.xhtml", media_type="application/xhtml+xml",
                        content=_nav_xhtml(language, toc, page_list).encode("utf-8"))
    nav.properties = ["nav"]
    book.add_item(nav)
    book.add_item(epub.EpubNcx())

    book.toc = tuple(epub.Link(f"{sec.id}.xhtml", sec.title, sec.id)
                     for sec in sections if sec.in_toc)
    book.spine = [nav, *spine_items]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    epub.write_epub(str(output_path), book)
    return output_path


def _media_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".gif": "image/gif", ".svg": "image/svg+xml"}.get(ext, "image/png")
