# epubocr — a fidelity-first OCR pipeline for image-only EPUBs

Many EPUBs are just scanned page images wrapped in XHTML. `epubocr` turns them into improved
EPUBs where **OCR is the source of truth** and any LLM/VLM only restructures the text — it never
re-transcribes the image. The design is **fidelity-first** (faithful text beats pretty text),
**measurable** (an eval harness decides engine/prompt choices), and **cache-first** (every stage is
reversible and re-runnable).

It runs across two machines: a laptop does the CPU/IO work plus local OCR; an optional GPU host
serves a VLM and an LLM over OpenAI-compatible HTTP endpoints.

## Pipeline

```text
EPUB
  ↓ unpack + inspect spine/manifest (zipfile + lxml)
  ↓ classify pages, flatten to one entry per page image (handles many-images-per-doc scans)
  ↓ extract page images in reading order  → hash + dedup
  ↓ preprocess (adaptive, measured — not a fixed chain)
  ↓ OCR  → per-page JSON (text + boxes + confidence)        [cache]   + degeneracy guard
  ↓ ── EVAL HARNESS gates engine choice against a small gold set ──
  ↓ deterministic layout cleanup (headers/footers, de-hyphenation, paragraph rejoin)
  ↓ LLM structure pass  → XHTML fragments        [behind a fidelity verifier]
  ↓ per-page assembly: reflowable prose  |  facsimile fallback for hard/low-confidence pages
  ↓ build EPUB3 (EbookLib) with page-list nav + pagebreak anchors
  ↓ validate (EPUBCheck + structural checks)
improved EPUB
```

Three principles run through every stage:

1. **OCR is ground truth.** The LLM restructures OCR output; it never re-transcribes the image.
2. **Cacheable & reversible.** Keep `original` + `processed` images and every intermediate; cache by
   `hash(content) + params + engine/model + prompt-version` so a re-run only redoes what changed.
3. **Measurable.** No engine, preprocessing step, or prompt ships without moving a number on the gold set.

## Topology

| Role                     | Machine        | Responsibilities                                                              |
| ------------------------ | -------------- | ----------------------------------------------------------------------------- |
| Orchestrator / CPU / IO  | Laptop         | EPUB unpack, image extract, preprocessing, EPUB build, validation, QA         |
| Local GPU                | Laptop         | Traditional OCR (Surya / Tesseract / Paddle)                                  |
| Remote GPU (optional)    | GPU host       | VLM-OCR + LLM cleanup via OpenAI-compatible endpoints (Ollama / vLLM)          |

Endpoints live in `config.toml` (copy to a gitignored `config.local.toml` for your real hosts):

| Endpoint | Purpose                                        | Notes                                                       |
| -------- | ---------------------------------------------- | ----------------------------------------------------------- |
| Ollama   | text cleanup (and on-demand VLM)               | swaps model per request — OCR and cleanup can interleave     |
| vLLM     | high-throughput VLM-OCR / batch cleanup        | one model per process; higher throughput                    |

If you front a relay, target the **IP literal** in config (not `localhost`) — IPv4-only relays make
Python's happy-eyeballs stall ~5 s on IPv6 first.

## OCR engines — fidelity-first policy

Traditional OCR is the default; the VLM is a *measured challenger* and a scalpel for hard regions —
never the blind default. Traditional OCR transcribes pixels deterministically: its errors are garbled
characters (detectable, and it reports low confidence). A generative VLM instead produces fluent,
plausible **hallucinations** — the dangerous failure mode.

| Engine        | Where    | Strengths                                | Watch-outs                                       |
| ------------- | -------- | ---------------------------------------- | ------------------------------------------------ |
| **Surya**     | Laptop   | Layout, reading order, tables, **confidence** | RAIL-M model license; tune batch size for ≤8 GB |
| Tesseract     | Laptop   | Trivial CPU baseline                     | Weak on complex layout / faint scans             |
| PaddleOCR     | Laptop   | Fast, light, Apache-2.0                  | Windows CUDA-wheel friction                      |
| Qwen2.5-VL    | GPU host | Hard pages, structure understanding      | Confabulates on faint text → verifier required   |

All sit behind an `OCREngine` interface so the eval harness compares them on equal footing.

**Observed in practice (faded scans):** the VLM produced fluent prose with *invented* proper nouns
(confident and wrong, no confidence signal); Surya produced garbled text with a **low confidence
score** that routes the page to facsimile + QA. That is exactly why Surya is the fidelity default.

### Two safety nets at the two model stages

- **OCR degeneracy guard.** VLMs occasionally fall into repetition loops (`[illegible] [illegible]…`).
  The verifier can't catch this (it guards cleanup, not OCR), so a `repetition_ratio` detector flags
  degenerate OCR and routes the page to facsimile instead of trusting it.
- **Fidelity verifier (cleanup).** After each LLM pass, the cleaned text is aligned to the OCR ground
  truth; if char edit-distance, inserted-word ratio, or length delta exceeds thresholds, the page is
  **held** and falls back to the deterministically-cleaned OCR. The prompt asks the model not to invent;
  this checks.

## Layout reconstruction — deterministic first

Do the mechanical parts deterministically (cheaper, no hallucination surface); reserve the LLM for
semantic structure.

| Task                      | Done with                                                       |
| ------------------------- | --------------------------------------------------------------- |
| Reading order             | the OCR/layout engine (Surya)                                   |
| Header/footer + page-no.  | deterministic repeated-text + position clustering across pages  |
| Hyphenation repair        | deterministic (`exam-\\nple` → `example`)                       |
| Chapter / footnote detect | LLM (semantic), behind the fidelity verifier                    |
| Tables / math             | specialized handling → often facsimile fallback                 |

## Output — per-page adaptive

Reflowable XHTML for prose; **facsimile fallback** (page image + OCR text layer) for tables, poetry,
math, illustrations, or any page below a confidence floor / held by the verifier. The EPUB3 preserves
print pagination via a `page-list` nav and inline `epub:type="pagebreak" role="doc-pagebreak"` anchors.

## Project layout

```
epubocr/
  config.py     # endpoints, per-endpoint model aliases, thresholds (prefers config.local.toml)
  ingest.py     # zipfile + lxml; per-image page model + classification
  preprocess.py # adaptive Pillow/OpenCV steps
  ocr/          # OCREngine interface + surya, tesseract, paddle, vlm_openai engines
  llm/          # OpenAI-compatible client + cleanup passes + fidelity_verifier
  layout.py     # deterministic header/footer removal, de-hyphenation, paragraph rejoin
  assemble.py   # per-page reflowable/facsimile XHTML; EbookLib build; page-list nav
  validate.py   # EPUBCheck wrapper + structural checks
  eval/         # gold set + metrics (CER/WER, insertion/hallucination, repetition)
  pipeline.py   # OCR + build orchestration
  storage.py    # book_project/ layout + content-addressed cache
  cli.py        # per-stage, resumable, cache-aware
```

## Usage

```bash
uv sync                                   # core deps (Python 3.12)
uv sync --extra surya                     # add Surya (pulls torch; --torch-backend=auto for CUDA)

uv run epubocr ingest path/to/book.epub   # → extracted/manifest.json, page images
uv run epubocr ocr  <book>                # OCR image pages (default engine: surya)
uv run epubocr eval <book> --gold g.json --engines surya,vlm    # compare engines on a gold set
uv run epubocr build <book> [--llm]       # assemble the improved EPUB (--llm runs cleanup)
uv run epubocr endpoints                  # live reachability check of configured endpoints
```

Per book, an intermediate `book_projects/<book>/` holds `source/`, `extracted/`, `ocr/`, `cleaned/`,
`qa/`, `cache/`, and `output/improved.epub` — OCR is expensive, so nothing is recomputed needlessly.

## Status

Working end-to-end: ingest + classification (incl. many-images-per-document scans), Surya and
VLM-OCR engines, the eval harness, deterministic + LLM cleanup with the fidelity verifier, and
per-page adaptive EPUB assembly with a page-list nav. EPUBCheck validation requires a JRE; Tesseract
and Paddle engines require their extras. The VLM remains a structure aid and challenger, not the
primary transcriber.
