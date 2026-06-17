"""EPUB ingestion + page classification (epubOCR.md §1-2).

Uses plain ``zipfile`` + ``lxml`` over the OPF. The unit of work is the **page
image**, not the spine document — image-only EPUBs come in two shapes and both
must flatten to an ordered list of pages:

  * one XHTML wrapper per page image (common converter output), and
  * one XHTML document embedding many page images (Calibre / Internet-Archive scans).

Produces ``extracted/manifest.json`` (one entry per page) and copies page images
into ``extracted/pages`` in reading order.
"""
from __future__ import annotations

import posixpath
import shutil
import zipfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from lxml import etree

from .storage import BookProject

_CONTAINER = "META-INF/container.xml"
_TEXT_MIN_CHARS = 12        # below this a page has no real text
_TEXT_PER_IMAGE = 200       # avg real chars/image above which a doc is prose-with-figures


class PageType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    MIXED = "mixed"
    COVER = "cover"
    EMPTY = "empty"


@dataclass
class SpinePage:
    index: int                      # global page index in reading order
    idref: str                      # source spine item id
    href: str                       # source XHTML doc (OPF-relative)
    page_type: PageType
    text_len: int
    images: list[str] = field(default_factory=list)            # source image hrefs (0 or 1)
    extracted_images: list[str] = field(default_factory=list)  # pages/ filenames (0 or 1)

    def to_json(self) -> dict:
        return {
            "index": self.index, "idref": self.idref, "href": self.href,
            "type": self.page_type.value, "text_len": self.text_len,
            "images": self.images, "extracted_images": self.extracted_images,
        }


def _localname(tag) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) and "}" in tag else tag


def _resolve(base_href: str, rel: str) -> str:
    return posixpath.normpath(posixpath.join(posixpath.dirname(base_href), rel))


def _opf_path(zf: zipfile.ZipFile) -> str:
    root = etree.fromstring(zf.read(_CONTAINER))
    for el in root.iter():
        if _localname(el.tag) == "rootfile":
            return el.get("full-path")
    raise ValueError("container.xml has no <rootfile>")


def _parse_opf(zf: zipfile.ZipFile, opf_path: str):
    root = etree.fromstring(zf.read(opf_path))
    manifest: dict[str, dict] = {}
    cover_id = None
    spine_ids: list[str] = []
    for el in root.iter():
        ln = _localname(el.tag)
        if ln == "item":
            manifest[el.get("id")] = {"href": el.get("href"), "media_type": el.get("media-type", ""),
                                      "properties": el.get("properties", "")}
        elif ln == "itemref":
            spine_ids.append(el.get("idref"))
        elif ln == "meta" and el.get("name") == "cover":
            cover_id = el.get("content")
    for iid, item in manifest.items():
        if "cover-image" in (item.get("properties") or ""):
            cover_id = cover_id or iid
    return manifest, spine_ids, cover_id


def _visible_text(el) -> str:
    """Text content excluding <style>/<script> subtrees (which aren't page text)."""
    if _localname(el.tag) in ("style", "script"):
        return ""
    parts = [el.text or ""]
    for child in el:
        parts.append(_visible_text(child))
        parts.append(child.tail or "")
    return "".join(parts)


def _doc_images(tree, doc_href: str) -> list[str]:
    """Resolved image hrefs in document (reading) order."""
    out: list[str] = []
    for el in tree.iter():
        ln = _localname(el.tag)
        if ln == "img" and el.get("src"):
            out.append(_resolve(doc_href, el.get("src")))
        elif ln == "image":  # SVG <image xlink:href|href>
            href = el.get("{http://www.w3.org/1999/xlink}href") or el.get("href")
            if href:
                out.append(_resolve(doc_href, href))
    return out


def _body(tree):
    for el in tree.iter():
        if _localname(el.tag) == "body":
            return el
    return tree


def ingest(epub_path: Path, project: BookProject) -> list[SpinePage]:
    project.ensure()
    epub_path = Path(epub_path)
    dest = project.source / epub_path.name
    if not dest.exists():
        shutil.copy2(epub_path, dest)

    pages: list[SpinePage] = []
    gidx = 0
    with zipfile.ZipFile(epub_path) as zf:
        opf = _opf_path(zf)
        manifest, spine_ids, cover_id = _parse_opf(zf, opf)
        names = set(zf.namelist())
        cover_hrefs: set[str] = set()
        if cover_id and cover_id in manifest:
            cover_hrefs.add(_resolve(opf, manifest[cover_id]["href"]))

        for sidpos, idref in enumerate(spine_ids):
            item = manifest.get(idref)
            if not item:
                continue
            href = _resolve(opf, item["href"])
            try:
                tree = etree.fromstring(zf.read(href), parser=etree.XMLParser(recover=True))
            except (KeyError, etree.XMLSyntaxError):
                continue

            text_len = len(" ".join(_visible_text(_body(tree)).split()))
            imgs = _doc_images(tree, href)

            # Prose-with-figures: lots of real text per image -> one preserved text page.
            if imgs and text_len < _TEXT_PER_IMAGE * len(imgs):
                for j, img in enumerate(imgs):
                    if img not in names:
                        continue
                    is_cover = img in cover_hrefs or (
                        gidx == 0 and len(imgs) == 1 and text_len < _TEXT_MIN_CHARS
                        and ("cover" in href.lower() or "title" in href.lower()))
                    ptype = PageType.COVER if is_cover else PageType.IMAGE
                    out_name = f"page_{gidx + 1:04d}_{Path(img).name}"
                    (project.pages / out_name).write_bytes(zf.read(img))
                    pages.append(SpinePage(gidx, idref, href, ptype,
                                           text_len if j == 0 else 0, [img], [out_name]))
                    gidx += 1
            elif text_len >= _TEXT_MIN_CHARS:
                ptype = PageType.MIXED if imgs else PageType.TEXT
                pages.append(SpinePage(gidx, idref, href, ptype, text_len, imgs, []))
                gidx += 1
            else:
                pages.append(SpinePage(gidx, idref, href, PageType.EMPTY, text_len, [], []))
                gidx += 1

    project.write_json(project.manifest_path, {
        "epub": epub_path.name,
        "opf": opf,
        "page_count": len(pages),
        "counts": {t.value: sum(1 for p in pages if p.page_type is t) for t in PageType},
        "pages": [p.to_json() for p in pages],
    })
    return pages
