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
from .assemble import OutputMode, Section, SpineDoc, blocks_to_xhtml, build_epub, paragraphs_to_xhtml
from .eval import metrics

# `mixed` pages (real text + a figure) are NOT OCR'd: they carry their own text and have
# no extracted page image, so they're preserved as text in build_book (real text > OCR).
_OCR_PAGE_TYPES = {"image", "cover"}
# Bump to invalidate the OCR stage cache. ocr-2: Surya 2 joins layout blocks with blank
# lines so paragraph boundaries survive into the EPUB (was a single '\n' -> one <p>/page).
_OCR_VERSION = "ocr-2"
# Pages handed to engine.run_batch per call. Bounds client memory + barrier granularity
# (and gives incremental cache writes so a crash mid-book keeps finished pages). The
# server-side concurrency within a chunk is the engine's own (SURYA_INFERENCE_PARALLEL).
_OCR_BATCH = 32


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


def _prep_image(src: Path, cfg: Config, use_pp: bool) -> Path:
    """Return the image path to OCR — preprocessed when the engine wants it, else src."""
    if not use_pp:
        return src
    processed = src.with_name(src.stem.replace("_original", "") + "_processed.png")
    return preprocess_image(src, processed, cfg.preprocess)


def ocr_image(engine: OCREngine, src: Path, project: BookProject, cfg: Config,
              preprocess: bool | None = None) -> tuple[OcrResult, Path]:
    """Preprocess (if requested) then OCR a single image. ``preprocess`` overrides the
    engine default (None -> engine.wants_preprocess). Returns (result, image_used)."""
    use_pp = engine.wants_preprocess if preprocess is None else preprocess
    image_used = _prep_image(src, cfg, use_pp)
    return engine.run(image_used), image_used


# The engine instance from the most recent ocr_book call, so the CLI can print a pool's
# per-unit throughput without ocr_book's return type having to know about pools.
_last_engine: dict = {}


def ocr_book(project: BookProject, engine_name: str, cfg: Config, *, force: bool = False,
             limit: int | None = None, preprocess: bool | None = None,
             batch_size: int = _OCR_BATCH) -> list[PageOcr]:
    """OCR a book's image pages, batching the cache-misses through ``engine.run_batch``.

    Three passes so the served-VLM/Surya engines OCR many pages concurrently instead of
    one HTTP round-trip at a time: (1) build the page work-list + cache keys and read any
    cache hits, (2) ``run_batch`` the misses in chunks (caching each), (3) write per-page
    raw.json/text.txt + the degeneracy guard, in spine order. Per-page caching, ``force``,
    ``limit``, and the routing are all unchanged — only the OCR calls are batched.
    """
    manifest = project.read_json(project.manifest_path)
    engine = get_engine(engine_name, cfg)
    _last_engine["engine"] = engine          # so the CLI can report pool per-unit stats
    use_pp = engine.wants_preprocess if preprocess is None else preprocess
    params = {"engine": engine.identity(), "preprocess": cfg.preprocess if use_pp else None}

    # Pass 1 — work-list: each image page with its cache key and any cache hit.
    work: list[dict] = []
    for page in manifest["pages"]:
        if page["type"] not in _OCR_PAGE_TYPES:
            continue
        if limit is not None and len(work) >= limit:
            break
        src = _image_for_page(project, page)
        if src is None or not src.exists():
            continue
        key = cache_key(content=src.read_bytes(), params=params, model=engine.identity(),
                        version=_OCR_VERSION)
        work.append({"page": page, "src": src, "key": key,
                     "cached": None if force else project.cache_get("ocr", key), "result": None})

    # Pass 2 — batch-OCR the cache-misses; the engine fans the chunk out concurrently.
    # A pooled engine asks for a bigger batch: its own split has to reach every unit, and
    # a batch <= its per-unit chunk would leave all but one unit idle (see PoolEngine).
    batch_size = max(batch_size, getattr(engine, "preferred_batch", 0))
    todo = [w for w in work if w["cached"] is None]
    for i in range(0, len(todo), max(1, batch_size)):
        chunk = todo[i:i + max(1, batch_size)]
        srcs = [_prep_image(w["src"], cfg, use_pp) for w in chunk]
        for w, result in zip(chunk, engine.run_batch(srcs)):
            project.cache_put("ocr", w["key"], result.to_json())
            w["result"] = result

    # Pass 3 — resolve every page's result, then post-process + write in spine order.
    for w in work:
        if w["cached"] is not None:
            c = w["cached"]
            w["result"] = OcrResult(text=c["text"], words=[], mean_conf=c.get("mean_conf"),
                                    engine=c.get("engine", engine.identity()),
                                    meta=c.get("meta", {}))
            w["was_cached"] = True
        else:
            w["was_cached"] = False

    # The length guard is book-relative, so it needs every page before judging any: a
    # degenerate page is one that ran long against THIS book's median, not an absolute size.
    outliers = metrics.length_outliers(
        {i: len(w["result"].text) for i, w in enumerate(work)},
        factor=float(cfg.raw.get("ocr", {}).get("degenerate_length_factor", 2.0)))

    out: list[PageOcr] = []
    for i, w in enumerate(work):
        page = w["page"]
        result = w["result"]

        # The fidelity verifier guards the cleanup pass, not OCR — so screen OCR output
        # here (epubOCR.md §4/§9). TWO tests, because one is not enough: repetition catches
        # a single token looping ('[illegible] [illegible]…'), while the length outlier
        # catches a model looping over VARIED filler or fabricating on a blank page — which
        # slips under the repetition threshold entirely (measured 2026-08-06, see fleet.md
        # §11). Either verdict blanks the text so the builder routes the page to facsimile.
        rep = metrics.repetition_ratio(result.text)
        rep_degenerate = metrics.is_degenerate(result.text)
        long_degenerate = i in outliers
        degenerate = rep_degenerate or long_degenerate
        text_out = "" if degenerate else result.text

        stem = f"page_{page['index'] + 1:04d}"
        raw = result.to_json()
        raw["repetition_ratio"] = round(rep, 3)
        raw["degenerate"] = degenerate
        if degenerate:
            raw["degenerate_reason"] = "repetition" if rep_degenerate else "length_outlier"
        project.write_json(project.ocr / f"{stem}.raw.json", raw)
        (project.ocr / f"{stem}.text.txt").write_text(text_out, encoding="utf-8")
        out.append(PageOcr(index=page["index"], engine=result.engine, mean_conf=result.mean_conf,
                           text_chars=len(text_out.strip()), cached=w["was_cached"],
                           degenerate=degenerate))
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


def _consensus_candidates(project: BookProject, manifest: dict, ocr_text_by_idx: dict[int, str],
                          conf_floor: float, budget: int) -> list[int]:
    """Page indices worth spending a challenger transcription on, least trusted first.

    The budget is spent where self-reported confidence is weakest — a page with **no**
    confidence at all (the VLM-family engines) ranks first, since for those the floor in
    ``build_book`` cannot fire and consensus is the *only* check besides the degeneracy
    guard. Pages already below the floor are skipped: they are going to facsimile anyway,
    so a challenger would change nothing.
    """
    scored: list[tuple[float, int]] = []
    for page in manifest["pages"]:
        if page["type"] != "image":
            continue
        idx = page["index"]
        if not ocr_text_by_idx.get(idx, "").strip():
            continue
        conf = _page_conf(project, f"page_{idx + 1:04d}")
        if conf is not None and conf < conf_floor:
            continue                       # already routed to facsimile
        scored.append((-1.0 if conf is None else conf, idx))
    scored.sort()
    return [idx for _c, idx in scored[:budget]]


def build_book(project: BookProject, cfg: Config, *, use_llm: bool = False,
               title: str | None = None, cleanup_endpoint: str | None = None,
               cleanup_model: str | None = None, conf_floor: float = 0.80,
               consensus_engine: str | None = None,
               consensus_max_pages: int = 0) -> tuple[Path, dict]:
    """Assemble an improved EPUB from the manifest + per-page OCR (deterministic by default).

    Per-page adaptive output (epubOCR.md §8): cover and low-confidence pages -> facsimile
    (with the OCR text kept as a hidden searchable layer); confident scanned pages ->
    reflowable from cleaned OCR; real text pages -> preserved. ``conf_floor`` is the OCR
    confidence below which a page is preserved as facsimile rather than trusted as text.

    ``consensus_engine`` turns on the cross-engine check (epubOCR.md §4): up to
    ``consensus_max_pages`` of the least-trusted reflowable candidates are re-transcribed
    by a second, independent engine and put through :func:`consensus.assess`; a page the
    two engines disagree on is routed to facsimile even if the primary was confident.
    This is the only guard that catches a *confident* fabrication — see fleet.md §11.
    """
    manifest = project.read_json(project.manifest_path)
    source_epub = next(iter(project.source.glob("*.epub")), None)

    # Pre-pass: gather per-page OCR text so running heads/footers/page numbers — lines that
    # recur across the book — can be detected once and stripped from every page (epubOCR.md
    # §6). Without this they leak into the reflowable prose (e.g. a bare page number "7").
    # Only `image` pages carry OCR text; `mixed`/`text` keep their real source text below.
    ocr_text_by_idx: dict[int, str] = {}
    for page in manifest["pages"]:
        if page["type"] == "image":
            tp = project.ocr / f"page_{page['index'] + 1:04d}.text.txt"
            if tp.exists():
                ocr_text_by_idx[page["index"]] = tp.read_text(encoding="utf-8")
    running = layout.find_running_lines([ocr_text_by_idx[i] for i in sorted(ocr_text_by_idx)])

    # Chapter starts from the embedded outline (PDF bookmarks / EPUB nav), recorded at ingest.
    chapter_titles = {int(e[0]): str(e[1]).strip()
                      for e in (manifest.get("outline") or []) if e and len(e) >= 2 and e[1]}
    book_title = title or Path(manifest.get("epub") or manifest.get("source_file") or "book").stem

    # Cross-engine consensus: transcribe the least-trusted candidates with a second engine
    # up front (batched, so it rides the same fan-out as OCR) and judge each page below.
    consensus_by_idx: dict[int, "consensus.Consensus"] = {}
    if consensus_engine and consensus_max_pages > 0:
        from . import consensus as _consensus

        cand = _consensus_candidates(project, manifest, ocr_text_by_idx, conf_floor,
                                     consensus_max_pages)
        if cand:
            challenger = get_engine(consensus_engine, cfg)
            by_idx = {p["index"]: p for p in manifest["pages"]}
            srcs, kept = [], []
            for idx in cand:
                src = _image_for_page(project, by_idx[idx])
                if src and src.exists():
                    srcs.append(_prep_image(src, cfg, challenger.wants_preprocess))
                    kept.append(idx)
            for idx, res in zip(kept, challenger.run_batch(srcs)):
                consensus_by_idx[idx] = _consensus.assess(
                    ocr_text_by_idx.get(idx, ""), _page_conf(project, f"page_{idx + 1:04d}"),
                    res.text, conf_floor=conf_floor)

    page_docs: list[tuple[SpineDoc, str | None]] = []   # (doc, chapter title if it opens one)
    page_no = 0
    counts = {"reflowable": 0, "facsimile": 0, "preserved": 0, "held": 0,
              "consensus_checked": len(consensus_by_idx),
              "consensus_rejected": 0}

    for page in manifest["pages"]:
        ptype, idx = page["type"], page["index"]
        if ptype == "empty":
            continue
        if ptype == "cover":  # cover -> its own section (no page number; flow doesn't run through it)
            page_docs.append((SpineDoc(idx, "Cover", OutputMode.FACSIMILE,
                                       image_path=_image_for_page(project, page)), "Cover"))
            counts["facsimile"] += 1
            continue

        page_no += 1
        ch_title = chapter_titles.get(idx)
        if ptype == "image":
            stem = f"page_{idx + 1:04d}"
            raw = layout.strip_running_lines(ocr_text_by_idx.get(idx, ""), running)
            conf = _page_conf(project, stem)
            blocks = _page_blocks(project, stem)
            # Fallback chapter signal: a page that opens with a layout heading (Surya layout=True).
            if ch_title is None and blocks and blocks[0].get("type") == "heading" and blocks[0].get("text"):
                ch_title = blocks[0]["text"]
            # Reflowable only with usable text AND adequate confidence. A traditional engine
            # below the floor (or no text) -> facsimile; a VLM (conf=None) stays reflowable and
            # relies on the degeneracy guard + fidelity verifier instead.
            low_conf = conf is not None and conf < conf_floor
            # A consensus rejection overrides a confident primary: self-reported confidence
            # is exactly what fails on a fabricated page (fleet.md §11).
            cons = consensus_by_idx.get(idx)
            if cons is not None and not cons.trusted:
                low_conf = True
                counts["consensus_rejected"] += 1
            if raw.strip() and not low_conf:
                body, held = _cleaned_body(cfg, raw, use_llm, cleanup_endpoint, cleanup_model, blocks)
                if held:
                    counts["held"] += 1
                doc = SpineDoc(idx, f"Page {page_no}", OutputMode.REFLOWABLE,
                               page_number=page_no, body_xhtml=body)
                counts["reflowable"] += 1
            else:  # low-confidence or un-OCR'd -> facsimile, keep OCR text as searchable layer
                doc = SpineDoc(idx, f"Page {page_no}", OutputMode.FACSIMILE,
                               page_number=page_no, image_path=_image_for_page(project, page),
                               ocr_text=raw.strip() or None)
                counts["facsimile"] += 1
            page_docs.append((doc, ch_title))
        elif ptype in ("text", "mixed"):
            # Real text beats OCR (epubOCR.md §2). A born-digital PDF page carries its text in
            # the manifest; an EPUB `text`/`mixed` page is re-read from source. `mixed` pages
            # (prose + a figure) have substantial real text and no extracted page image, so they
            # are preserved here rather than routed to OCR — which would emit a blank page.
            body = page.get("text_html") or (
                _preserve_text_body(source_epub, page["href"]) if source_epub else "<p></p>")
            page_docs.append((SpineDoc(idx, f"Page {page_no}", OutputMode.REFLOWABLE,
                                       page_number=page_no, body_xhtml=body), ch_title))
            counts["preserved"] += 1

    sections = _group_sections(page_docs, book_title)
    out_path = build_epub(sections, title=book_title,
                          output_path=project.output / "improved.epub", stitch=not use_llm)
    return out_path, {"docs": sum(len(s.docs) for s in sections), "sections": len(sections), **counts}


def _group_sections(page_docs: list[tuple[SpineDoc, str | None]], book_title: str) -> list[Section]:
    """Group per-page docs into spine documents so the text flows: a new Section at the cover
    and at each chapter start (a page carrying a chapter title); consecutive pages otherwise
    flow into one Section. With no chapter signal the whole body becomes a single flowing
    Section (one document = no per-page break), which is what a book with no chapters gets.
    """
    sections: list[Section] = []

    def open_section(t: str) -> Section:
        sec = Section(id=f"sec{len(sections) + 1:04d}", title=t or book_title)
        sections.append(sec)
        return sec

    cur: Section | None = None
    for doc, ch_title in page_docs:
        if doc.page_number is None:                 # cover / front-matter image -> standalone
            open_section(ch_title or "Cover").docs.append(doc)
            cur = None                              # force the next content page to open fresh
            continue
        if cur is None or ch_title is not None:     # chapter boundary -> new flowing document
            cur = open_section(ch_title or book_title)
        cur.docs.append(doc)
    return sections or [Section(id="sec0001", title=book_title)]


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
