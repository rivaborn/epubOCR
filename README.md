# epubocr

**A fidelity-first OCR pipeline that turns image-only (scanned) EPUBs into improved, reflowable
EPUBs — without letting an LLM hallucinate the text.**

Many EPUBs are just page scans wrapped in XHTML. `epubocr` re-OCRs them with **OCR as the source of
truth**, uses an LLM only to *restructure* (never to re-transcribe), and falls back to the original
page image wherever it isn't confident.

## Why fidelity-first?

On a real faded scan, a vision-LLM and a traditional OCR engine fail in opposite — and instructive —
ways:

| Behavior on a faint page | Qwen2.5-VL                       | Surya                         |
| ------------------------ | -------------------------------- | ----------------------------- |
| Output                   | fluent prose with **invented names** | garbled text              |
| Confidence signal        | none                             | low (~0.45) — flagged         |
| Failure mode             | **silent** (confident & wrong)   | **loud** (routed to facsimile) |

A generative VLM confabulates *confidently*; traditional OCR fails *visibly* and reports a low
confidence you can act on. So `epubocr` defaults to traditional OCR (Surya) with confidence, treats
the VLM as a measured challenger, and routes low-confidence or degenerate pages to facsimile instead
of trusting them.

## Features

- **Pluggable OCR engines** behind one interface — Surya (default), Qwen2.5-VL via an
  OpenAI-compatible endpoint, Tesseract, PaddleOCR.
- **Eval harness** — CER / WER / insertion against a small gold set picks the engine empirically,
  not by opinion.
- **Fidelity verifier** — holds any LLM cleanup that drifts from the OCR ground truth (char
  edit-distance, inserted-word ratio, length delta) and falls back to the faithful text.
- **OCR degeneracy guard** — detects VLM repetition loops and routes those pages to facsimile.
- **Per-page adaptive output** — reflowable XHTML for prose, facsimile fallback for tables, poetry,
  math, illustrations, or low-confidence pages; EPUB3 `page-list` nav + pagebreak anchors.
- **Cache-first** — every stage keyed on `content + params + model + prompt-version`; re-runs only
  redo what changed.
- Handles both EPUB shapes: one-XHTML-wrapper-per-image, and many-images-per-document scans.

## Quick start

```bash
uv sync                            # core deps (Python 3.12)
uv sync --extra surya              # Surya engine (pulls torch; UV_TORCH_BACKEND/--torch-backend=auto for CUDA)
cp config.toml config.local.toml   # set your real LLM/VLM endpoints here (only needed for VLM/LLM stages)

uv run epubocr ingest book.epub    # → book_projects/book/extracted/manifest.json + page images
uv run epubocr ocr   book          # OCR image pages (default engine: surya)
uv run epubocr build book          # → book_projects/book/output/improved.epub
```

Other commands: `eval` (compare engines on a gold set), `endpoints` (live reachability check),
`ocr-page` (single-image smoke test), `show-config`.

## How it works

```
EPUB → unpack + classify → extract page images (hash+dedup) → adaptive preprocess
     → OCR (+ confidence, degeneracy guard) → [eval gate] → deterministic cleanup
     → LLM structure pass [behind fidelity verifier] → per-page reflowable/facsimile
     → EPUB3 (EbookLib, page-list nav) → EPUBCheck + structural checks
```

OCR is ground truth; the LLM only restructures; every stage is cacheable and reversible. See
[`epubOCR.md`](epubOCR.md) for the full design and rationale.

## Requirements

- **Python 3.12** (PyTorch has no CUDA wheels for 3.14).
- Optional: a GPU host serving **Ollama** and/or **vLLM** (OpenAI-compatible) for VLM-OCR and LLM
  cleanup; a **JRE** for EPUBCheck; the `ocr-local` (Tesseract) or `paddle` extras for those engines.

## Status

Working end-to-end on real books: ingest + classification, Surya and VLM-OCR engines, the eval
harness, deterministic + LLM cleanup with the fidelity verifier, and per-page adaptive EPUB
assembly. The VLM stays a structure aid and challenger, not the primary transcriber.

## License

No license file yet — add one before relying on this for anything but personal use. Note that the
Surya model weights carry their own (AI-Pubs RAIL-M) license.
