# epubocr

**A fidelity-first OCR pipeline that turns image-only (scanned) EPUBs and PDFs into improved,
reflowable EPUBs — without letting an LLM hallucinate the text.**

Many EPUBs (and PDFs) are just page scans. `epubocr` re-OCRs the scanned pages with **OCR as the
source of truth**, uses an LLM only to *restructure* (never to re-transcribe), preserves any real
(born-digital) text it finds, and falls back to the original page image wherever it isn't confident.

## Why fidelity-first?

On a real faded scan, a vision-LLM and a traditional OCR engine fail in opposite — and instructive —
ways:

| Behavior on a faint page | Qwen2.5-VL                       | Surya                         |
| ------------------------ | -------------------------------- | ----------------------------- |
| Output                   | fluent prose with **invented names** | garbled text              |
| Confidence signal        | none                             | low (~0.45) — flagged         |
| Failure mode             | **silent** (confident & wrong)   | **loud** (routed to facsimile) |

A *general-purpose* VLM (Qwen2.5-VL) confabulates *confidently*; purpose-built OCR fails *visibly* and
reports a confidence you can act on. The default engine is **Surya 2** — a purpose-built OCR VLM that,
validated on a real scan, behaves on the right side of that table: it reports per-page confidence and
returns **empty (not invented) text** on pages it can't read, so they route to facsimile. **Surya 0.17**
(`--engine surya`) is the fast, no-backend local alternative; the general `--engine vlm` is a measured
challenger, never the blind default. Low-confidence or degenerate pages always fall back to facsimile.

## Features

- **Pluggable OCR engines** behind one interface — **Surya 2** (default, served-VLM; needs a
  llama.cpp/vLLM backend), Surya 0.17 (fast, local, no backend), Qwen2.5-VL via an OpenAI-compatible
  endpoint, Tesseract, and PaddleOCR.
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
- **PDF input too** (`--extra pdf`) — born-digital text layers are preserved; scanned pages are
  lossless-extracted (or rendered at 300 DPI) and OCR'd. Same source-agnostic pipeline, same EPUB out.

## Quick start

```bash
uv sync                            # core deps (Python 3.12)
uv sync --extra surya2             # DEFAULT engine: Surya 2 (>=0.20, served-VLM — also needs a backend, below)
#  ...or `--extra surya` for the fast local Surya 0.17 (pulls torch; --torch-backend=auto for CUDA; not both)
uv sync --extra pdf                # PDF input (PyMuPDF — note: AGPL-3.0)
cp config.toml config.local.toml   # set your real LLM/VLM endpoints here (only needed for VLM/LLM stages)

# Surya 2 needs a backend: set LLAMA_CPP_BINARY to a llama-server binary (or point at a vLLM endpoint).
# No backend handy? Use the fast local engine instead: `epubocr ocr book --engine surya`.
uv run epubocr ingest book.epub    # (or book.pdf) → book_projects/book/extracted/manifest.json + page images
uv run epubocr ocr   book          # OCR image pages (default engine: surya2; --engine surya for the fast path)
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
  cleanup; a **JRE** for EPUBCheck; the `ocr-local` (Tesseract) or `paddle` extras for those engines;
  the `pdf` extra (**PyMuPDF — AGPL-3.0**) for PDF input.

## Status

Working end-to-end on real books: ingest + classification, Surya and VLM-OCR engines, the eval
harness, deterministic + LLM cleanup with the fidelity verifier, and per-page adaptive EPUB
assembly. The VLM stays a structure aid and challenger, not the primary transcriber.

## License

No license file yet — add one before relying on this for anything but personal use. Note that the
Surya model weights carry their own (AI-Pubs RAIL-M) license.
