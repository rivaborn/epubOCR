"""Stage orchestration over an ingested book (epubOCR.md §4, §10).

Currently: the OCR stage. Walks the manifest's image pages, optionally preprocesses,
runs the chosen engine, and writes per-page OCR JSON + text — all content-addressed
so re-runs only redo changed pages.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .ocr import get_engine
from .ocr.base import OCREngine, OcrResult
from .preprocess import preprocess_image
from .storage import BookProject, cache_key
from . import layout
from .assemble import OutputMode, SpineDoc, blocks_to_xhtml, build_epub, paragraphs_to_xhtml
from .eval import metrics

_OCR_PAGE_TYPES = {"image", "cover", "mixed"}
_OCR_VERSION = "ocr-1"


@dataclass
class PageOcr:
    index: int
    engine: str
    mean_conf: float | None
    text_chars: int
    cached: bool
    degenerate: bool = False


def _image_for_page(project: BookProject, page: dict) -> Path | None:
    imgs = page.get("extracted_images") or []
    if not imgs:
        return None
    return project.pages / imgs[0]


def ocr_image(engine: OCREngine, src: Path, project: BookProject, cfg: Config,
              preprocess: bool | None = None) -> tuple[OcrResult, Path]:
    """Preprocess (if requested) then OCR a single image. ``preprocess`` overrides the
    engine default (None -> engine.wants_preprocess). Returns (result, image_used)."""
    use_pp = engine.wants_preprocess if preprocess is None else preprocess
    image_used = src
    if use_pp:
        processed = src.with_name(src.stem.replace("_original", "") + "_processed.png")
        image_used = preprocess_image(src, processed, cfg.preprocess)
    return engine.run(image_used), image_used


def ocr_book(project: BookProject, engine_name: str, cfg: Config, *, force: bool = False,
             limit: int | None = None, preprocess: bool | None = None) -> list[PageOcr]:
    manifest = project.read_json(project.manifest_path)
    engine = get_engine(engine_name, cfg)
    use_pp = engine.wants_preprocess if preprocess is None else preprocess
    params = {"engine": engine.identity(), "preprocess": cfg.preprocess if use_pp else None}

    out: list[PageOcr] = []
    for page in manifest["pages"]:
        if page["type"] not in _OCR_PAGE_TYPES:
            continue
        if limit is not None and len(out) >= limit:
            break
        src = _image_for_page(project, page)
        if src is None or not src.exists():
            continue

        key = cache_key(content=src.read_bytes(), params=params, model=engine.identity(), version=_OCR_VERSION)
        cached = None if force else project.cache_get("ocr", key)
        if cached is not None:
            result = OcrResult(text=cached["text"], words=[], mean_conf=cached.get("mean_conf"),
                               engine=cached.get("engine", engine.identity()), meta=cached.get("meta", {}))
            was_cached = True
        else:
            result, _ = ocr_image(engine, src, project, cfg, preprocess=preprocess)
            project.cache_put("ocr", key, result.to_json())
            was_cached = False

        # The fidelity verifier guards the cleanup pass, not OCR — so screen OCR output
        # here for degenerate repetition loops (epubOCR.md §4/§9). Degenerate pages get
        # empty text so the builder routes them to facsimile + QA instead of trusting them.
        rep = metrics.repetition_ratio(result.text)
        degenerate = metrics.is_degenerate(result.text)
        text_out = "" if degenerate else result.text

        stem = f"page_{page['index'] + 1:04d}"
        raw = result.to_json()
        raw["repetition_ratio"] = round(rep, 3)
        raw["degenerate"] = degenerate
        project.write_json(project.ocr / f"{stem}.raw.json", raw)
        (project.ocr / f"{stem}.text.txt").write_text(text_out, encoding="utf-8")
        out.append(PageOcr(index=page["index"], engine=result.engine, mean_conf=result.mean_conf,
                           text_chars=len(text_out.strip()), cached=was_cached, degenerate=degenerate))
    return out


# ---------------------------------------------------------------------------
# Build stage: ingested + OCR'd project -> improved EPUB (epubOCR.md §7-8)
# ---------------------------------------------------------------------------
_BLOCK_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "blockquote", "li"}


def _localname(tag) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) and "}" in tag else tag


def _preserve_text_body(source_epub: Path, href: str) -> str:
    """Re-extract block elements (headings/paragraphs) from an original text page.

    Lossy on CSS/inline markup but preserves structure — enough to keep real EPUB
    text rather than OCR'ing it (epubOCR.md §2: real text beats OCR).
    """
    import html
    import zipfile

    from lxml import etree

    try:
        with zipfile.ZipFile(source_epub) as zf:
            xhtml = zf.read(href)
    except (KeyError, OSError):
        return "<p></p>"
    tree = etree.fromstring(xhtml, parser=etree.XMLParser(recover=True))
    parts: list[str] = []
    for el in tree.iter():
        if _localname(el.tag) in _BLOCK_TAGS:
            text = " ".join("".join(el.itertext()).split())
            if not text:
                continue
            tag = _localname(el.tag)
            tag = tag if tag in ("h1", "h2", "h3") else "p"
            parts.append(f"<{tag}>{html.escape(text)}</{tag}>")
    return "\n".join(parts) or "<p></p>"


def _page_conf(project: BookProject, stem: str) -> float | None:
    """Mean OCR confidence for a page, from its raw.json (None if the engine has none)."""
    p = project.ocr / f"{stem}.raw.json"
    return project.read_json(p).get("mean_conf") if p.exists() else None


def _page_blocks(project: BookProject, stem: str) -> list | None:
    """Layout blocks for a page, if the OCR engine produced them (Surya layout=True)."""
    p = project.ocr / f"{stem}.raw.json"
    if not p.exists():
        return None
    return (project.read_json(p).get("meta") or {}).get("blocks")


def build_book(project: BookProject, cfg: Config, *, use_llm: bool = False,
               title: str | None = None, cleanup_endpoint: str | None = None,
               cleanup_model: str | None = None, conf_floor: float = 0.80) -> tuple[Path, dict]:
    """Assemble an improved EPUB from the manifest + per-page OCR (deterministic by default).

    Per-page adaptive output (epubOCR.md §8): cover and low-confidence pages -> facsimile
    (with the OCR text kept as a hidden searchable layer); confident scanned pages ->
    reflowable from cleaned OCR; real text pages -> preserved. ``conf_floor`` is the OCR
    confidence below which a page is preserved as facsimile rather than trusted as text.
    """
    manifest = project.read_json(project.manifest_path)
    source_epub = next(iter(project.source.glob("*.epub")), None)

    docs: list[SpineDoc] = []
    page_no = 0
    counts = {"reflowable": 0, "facsimile": 0, "preserved": 0, "held": 0}

    for page in manifest["pages"]:
        ptype, idx = page["type"], page["index"]
        if ptype == "empty":
            continue
        if ptype == "cover":
            docs.append(SpineDoc(idx, "Cover", OutputMode.FACSIMILE,
                                 image_path=_image_for_page(project, page)))
            counts["facsimile"] += 1
            continue

        page_no += 1
        if ptype in ("image", "mixed"):
            stem = f"page_{idx + 1:04d}"
            text_path = project.ocr / f"{stem}.text.txt"
            raw = text_path.read_text(encoding="utf-8") if text_path.exists() else ""
            conf = _page_conf(project, stem)
            # Reflowable only with usable text AND adequate confidence. A traditional engine
            # below the floor (or no text) -> facsimile; a VLM (conf=None) stays reflowable and
            # relies on the degeneracy guard + fidelity verifier instead.
            low_conf = conf is not None and conf < conf_floor
            if raw.strip() and not low_conf:
                blocks = _page_blocks(project, stem)
                body, held = _cleaned_body(cfg, raw, use_llm, cleanup_endpoint, cleanup_model, blocks)
                if held:
                    counts["held"] += 1
                docs.append(SpineDoc(idx, f"Page {page_no}", OutputMode.REFLOWABLE,
                                     page_number=page_no, body_xhtml=body))
                counts["reflowable"] += 1
            else:  # low-confidence or un-OCR'd -> facsimile, keep OCR text as searchable layer
                docs.append(SpineDoc(idx, f"Page {page_no}", OutputMode.FACSIMILE,
                                     page_number=page_no, image_path=_image_for_page(project, page),
                                     ocr_text=raw.strip() or None))
                counts["facsimile"] += 1
        elif ptype == "text":
            # PDF born-digital text travels in the manifest; EPUB text is re-read from source.
            body = page.get("text_html") or (
                _preserve_text_body(source_epub, page["href"]) if source_epub else "<p></p>")
            docs.append(SpineDoc(idx, f"Page {page_no}", OutputMode.REFLOWABLE,
                                 page_number=page_no, body_xhtml=body))
            counts["preserved"] += 1

    out_path = build_epub(
        docs, title=title or Path(manifest.get("epub") or manifest.get("source_file") or "book").stem,
        output_path=project.output / "improved.epub")
    return out_path, {"docs": len(docs), **counts}


def _cleaned_body(cfg: Config, ocr_text: str, use_llm: bool,
                  cleanup_endpoint: str | None = None, cleanup_model: str | None = None,
                  blocks: list | None = None) -> tuple[str, bool]:
    """Deterministic cleanup, optionally an LLM structure pass behind the verifier.

    When Surya layout blocks are present they drive the deterministic structure
    (headings/lists/tables/reading order); otherwise fall back to paragraph-joining.
    """
    deterministic = blocks_to_xhtml(blocks) if blocks else paragraphs_to_xhtml(layout.clean_page_text(ocr_text))
    if not use_llm:
        return deterministic, False
    from .llm.cleanup import clean_page
    result = clean_page(cfg, ocr_text, deterministic,
                        endpoint_name=cleanup_endpoint, model=cleanup_model)
    return result.xhtml, result.used_fallback
