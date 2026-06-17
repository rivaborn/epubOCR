"""Generate a tiny image-only EPUB fixture for developing/verifying the pipeline.

Builds: a cover (image), two full-page image pages, and one real-text page — a mix
that exercises page classification. Run: ``uv run python tests/fixtures/make_sample_epub.py``
-> writes ``tests/fixtures/sample.epub``.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
OUT = HERE / "sample.epub"


def _page_png(text: str, size=(800, 1100)) -> bytes:
    img = Image.new("L", size, color=255)
    d = ImageDraw.Draw(img)
    for i, line in enumerate(text.splitlines()):
        d.text((60, 80 + i * 34), line, fill=20)
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _wrapper(img_href: str, title: str) -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>'
        f"{title}</title></head><body>"
        f'<img src="{img_href}" alt="{title}"/></body></html>'
    )


CONTAINER = (
    '<?xml version="1.0"?>\n'
    '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
    '<rootfiles><rootfile full-path="OEBPS/content.opf" '
    'media-type="application/oebps-package+xml"/></rootfiles></container>'
)

TEXT_PAGE = (
    '<?xml version="1.0" encoding="utf-8"?>\n'
    '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Chapter 1</title></head>'
    "<body><h1>Chapter 1</h1><p>This page already contains real, selectable text, so the "
    "classifier should mark it TEXT and the pipeline should preserve it rather than OCR it.</p>"
    "</body></html>"
)

OPF = (
    '<?xml version="1.0" encoding="utf-8"?>\n'
    '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid">'
    '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
    '<dc:identifier id="bookid">urn:uuid:epubocr-sample</dc:identifier>'
    "<dc:title>epubocr sample</dc:title><dc:language>en</dc:language>"
    '<meta name="cover" content="img-cover"/></metadata>'
    "<manifest>"
    '<item id="cover" href="cover.xhtml" media-type="application/xhtml+xml"/>'
    '<item id="p1" href="page_001.xhtml" media-type="application/xhtml+xml"/>'
    '<item id="p2" href="page_002.xhtml" media-type="application/xhtml+xml"/>'
    '<item id="t1" href="text_001.xhtml" media-type="application/xhtml+xml"/>'
    '<item id="img-cover" href="images/cover.png" media-type="image/png" properties="cover-image"/>'
    '<item id="img-p1" href="images/page_001.png" media-type="image/png"/>'
    '<item id="img-p2" href="images/page_002.png" media-type="image/png"/>'
    "</manifest>"
    '<spine><itemref idref="cover"/><itemref idref="p1"/>'
    '<itemref idref="p2"/><itemref idref="t1"/></spine>'
    "</package>"
)


def build() -> Path:
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml", CONTAINER)
        zf.writestr("OEBPS/content.opf", OPF)
        zf.writestr("OEBPS/cover.xhtml", _wrapper("images/cover.png", "Cover"))
        zf.writestr("OEBPS/page_001.xhtml", _wrapper("images/page_001.png", "Page 1"))
        zf.writestr("OEBPS/page_002.xhtml", _wrapper("images/page_002.png", "Page 2"))
        zf.writestr("OEBPS/text_001.xhtml", TEXT_PAGE)
        zf.writestr("OEBPS/images/cover.png", _page_png("epubocr\nsample\ncover"))
        zf.writestr("OEBPS/images/page_001.png", _page_png("Page one scanned image.\nThe quick brown fox\njumps over the lazy dog."))
        zf.writestr("OEBPS/images/page_002.png", _page_png("Page two scanned image.\nLorem ipsum dolor sit amet,\nconsectetur adipiscing elit."))
    return OUT


if __name__ == "__main__":
    print("wrote", build())
