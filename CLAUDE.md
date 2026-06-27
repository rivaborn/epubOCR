# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`epubocr` turns image-only (scanned) EPUBs **and PDFs** into improved, per-page-adaptive EPUBs. The governing
idea — read [`epubOCR.md`](epubOCR.md) for the full rationale — is **fidelity-first**: OCR output is
the ground truth, the LLM only *restructures* it (never re-transcribes), and any page the tool can't
read confidently falls back to a **facsimile** (page image + hidden searchable OCR layer) instead of
emitting hallucinated text.

## Commands

```bash
uv sync                              # core CLI deps (light, Python 3.12)
uv sync --extra surya2               # DEFAULT engine: Surya 2 (>=0.20) served-VLM — also needs a backend (below)
uv sync --extra surya                # fast `--engine surya` path: local Surya 0.17 (pulls torch — see CUDA note)
uv sync --extra ocr-local            # Tesseract + OpenCV (CPU preprocessing/deskew/binarize)
uv sync --extra pdf                  # PDF input via PyMuPDF (AGPL-3.0) — see gotchas
cp config.toml config.local.toml     # then put REAL endpoints here (gitignored; see Config)

# Pipeline (each stage is separate, resumable, cache-aware). <book> = a slug under
# book_projects/ OR a path to an .epub/.pdf (ingest takes the file; later stages the slug).
uv run epubocr ingest <book.epub|book.pdf>   # → book_projects/<slug>/extracted/manifest.json + page images
uv run epubocr ocr    <slug>         # OCR image pages → ocr/page_XXXX.{raw.json,text.txt}
uv run epubocr build  <slug>         # → book_projects/<slug>/output/improved.epub

# ocr flags:  --engine surya|surya2|vlm|tesseract|paddle  --limit N (sample)  --force (ignore cache)
#             --preprocess/--no-preprocess (override the engine's wants_preprocess default)
# build flags: --llm (run the LLM structure pass behind the fidelity verifier)  --title T
#              --cleanup-endpoint / --cleanup-model (reuse an already-loaded model)

uv run epubocr eval  <slug> --gold gold.json --engines surya,vlm   # CER/WER/insertion table
uv run epubocr ocr-page <image>      # single-image smoke test (no project needed)
uv run epubocr endpoints             # live reachability of each configured endpoint
uv run epubocr show-config           # resolved endpoints/roles/models/thresholds

# Test fixture + manual bake-off (there is NO pytest suite — see Testing):
uv run python tests/fixtures/make_sample_epub.py        # writes tests/fixtures/sample.epub
.venv/Scripts/python.exe scripts/bakeoff.py <slug> [n_surya] [n_vlm]
```

**CUDA torch gotcha:** `uv sync --extra surya` installs the **CPU** torch wheel. `uv sync` does not
honor `--torch-backend`, so swap in the CUDA build explicitly after:
`uv pip install --torch-backend=auto --reinstall-package torch "torch==2.12.0"` (verify with
`python -c "import torch; print(torch.__version__)"` → expect a `+cuXXX` suffix). Python is pinned to
3.12 because torch has no CUDA wheels for 3.14.

## Architecture

**Stage flow** (mirrors `epubocr/__init__.py`; every stage is content-addressed and reversible):

```
ingest → preprocess → ocr → (eval gate) → layout cleanup → llm cleanup → assemble → validate
```

| Module                      | Role                                                                                           |
| --------------------------- | ---------------------------------------------------------------------------------------------- |
| `ingest.py`                 | `zipfile`+`lxml` over the OPF → `manifest.json`; classifies & extracts **page images**         |
| `ingest_pdf.py`             | PDF front-end (PyMuPDF): same manifest — preserves text layers, extracts/renders scans         |
| `ocr/` (`base.OCREngine`)   | Pluggable engines: `surya_marker`, `surya2`, `vlm_openai`, `tesseract`, `paddle`               |
| `structure.py`              | Surya layout boxes + lines → ordered semantic `Block`s (opt-in; headings/lists/tables)         |
| `preprocess.py`             | Adaptive (conditional) Pillow/OpenCV steps — off by default; over-processing clean scans hurts |
| `layout.py`                 | Deterministic cleanup: de-hyphenate, rejoin paragraphs, strip running heads/footers            |
| `llm/`                      | OpenAI-compatible `client`, `cleanup` passes, and the `fidelity_verifier`                      |
| `assemble.py`               | Per-page reflowable/facsimile XHTML → EPUB3 via EbookLib (`page-list` nav, pagebreak anchors)  |
| `pipeline.py`               | `ocr_book` (batches cache-misses via `engine.run_batch`) and `build_book` — orchestration       |
| `eval/`                     | `metrics` (CER/WER/insertion/repetition) + `harness` (engine comparison table)                 |
| `consensus.py`              | Cross-engine agreement — a model-agnostic trust signal                                         |
| `storage.py`                | `BookProject` folder layout + the content-addressed `cache_key`                                |
| `config.py` / `validate.py` | Config loading; EPUBCheck + structural checks                                                  |

**The unit of work is the page image, not the spine document, and everything after ingest is
source-agnostic.** `ingest.py` flattens **two** EPUB shapes into one ordered page list: (a) one XHTML
wrapper per page image (converter output), and (b) one XHTML embedding many images (Calibre /
Internet-Archive scans). `ingest_pdf.py` (PyMuPDF) emits the **same** `manifest.json` from a PDF —
born-digital pages → `text` (the real text layer preserved in a `text_html` manifest field), image
pages → `image`. Pages are classified `text | image | mixed | cover | empty`; only `image/cover`
get OCR'd — a `mixed` page (real text + a figure) keeps no extracted image, so it is **preserved as
text** in `build_book` (real text beats OCR), not routed through the OCR'er (which would blank it).
OCR, the guards, per-page routing, and EPUB assembly never look at the original container, so the PDF
path reuses them unchanged.

**Fidelity is enforced by three distinct guards at different stages — do not conflate them:**
1. **OCR-stage degeneracy guard** (`eval/metrics.is_degenerate` via `repetition_ratio`, applied in
   `pipeline.ocr_book`): catches VLM repetition loops (`[illegible] [illegible]…`). A degenerate page
   gets its text blanked so the builder routes it to facsimile.
2. **Cleanup-stage fidelity verifier** (`llm/fidelity_verifier.verify`, applied in `llm/cleanup`):
   compares LLM XHTML against the OCR text (CER / inserted-word ratio / length delta vs `[fidelity]`
   thresholds). On drift it **discards the LLM output and falls back** to deterministic cleanup.
3. **Cross-engine consensus** (`consensus.assess`): trust a page where two independent engines agree.

**Per-page adaptive routing** lives in `pipeline.build_book` (`conf_floor=0.80`). A scanned page is
emitted **reflowable** only with usable text AND adequate confidence; otherwise **facsimile** (image +
hidden OCR text as a searchable layer). Subtlety: a traditional engine (and, as it turns out, Surya 2)
reports `mean_conf`, but the `vlm` engine reports `conf=None` — a `None`-confidence page stays reflowable
and relies on the degeneracy guard + fidelity verifier instead of the floor. Surya 2 returns real
per-block confidence on readable pages and `conf=None` + empty text on pages it can't read (covers,
near-blank), which the floor/empty-text check routes to facsimile. `text` pages are preserved from the
source (never re-OCR'd).

**Config indirection is three levels:** `endpoints` (URLs) → `roles` (`vlm_ocr`, `text_cleanup` → an
endpoint) → `models` (per-endpoint aliases, because Ollama tags `qwen2.5vl:7b` and vLLM served-names
`qwen2.5-vl-7b` differ). Resolve via `cfg.role_endpoint(role)` and `cfg.model(alias, endpoint=…)`.

**Caching:** `storage.cache_key(content, params, model, version)` keys every stage on content hash +
params + model + a stage/prompt version constant (`_OCR_VERSION`, `cleanup.PROMPT_VERSION`). Bump the
version constant to invalidate a stage; editing a prompt re-runs only that stage, never OCR.

## Config & endpoints

`config.py` prefers **`config.local.toml`** (gitignored, holds the real homelab IPs) over the
committed **`config.toml`** template (`127.0.0.1` placeholders); override with `$EPUBOCR_CONFIG`. Keep
private endpoints out of the committed template. From Python, target the **IP literal, not
`localhost`** — the vLLM relay is IPv4-only and the stdlib stalls ~5s trying IPv6 first. Endpoints are
optional: only the VLM-OCR (`--engine vlm`) and `--llm` cleanup stages call them; Surya/Tesseract and
the default build run fully local.

## Conventions & gotchas

- **Heavy backends are opt-in extras** (`surya`, `surya2`, `ocr-local`, `paddle`, `pdf`) so the core
  CLI installs fast; engines are lazy-imported in `ocr/__init__.get_engine`. Add deps to the right extra.
- **EbookLib:** `assemble.build_epub` uses **raw `EpubItem`** (not `EpubHtml`) for page docs *and* the
  nav, so our exact XHTML (pagebreak anchors, facsimile `<img>`, `epub:` namespace, custom `page-list`)
  is written verbatim — an `nav`-flagged `EpubHtml` gets templated away to empty.
- **Surya (local 0.17):** predictors share one `FoundationPredictor`; `layout=True` (opt-in via
  `[ocr] layout`) adds structured blocks but degrades on faded scans (`structure.build_blocks` walks
  lines in reading order, degrading gracefully to paragraph-joining). Strips inline `<b>`/`<i>`/`<math>`.
- **Two Surya engines, mutually exclusive; `surya2` is the default.** `surya2` (`surya2.Surya2Engine`)
  = **Surya 2 (≥0.20)**, a served-VLM needing a vllm/llama.cpp backend (`[ocr] surya2_backend`). Two
  ways to feed it: a local `llama-server` binary via `LLAMA_CPP_BINARY` (offline, on the 4060), or —
  the validated default on this setup — `surya2_backend = "vllm"` + `surya2_inference_url` +
  `surya2_model` to **attach** to an already-running vLLM (the homelab 3090 relay serves `surya-ocr-2`;
  load it with `llmconfig load vllm surya2`). Surya's attach path rejects a served-name mismatch, so
  `surya2_model` must equal the server's `--served-model-name`. The vLLM path is faster per page and
  reads at full bf16 precision; the engine sets these on Surya's settings singleton in `_ensure_loaded`. It is the default on **fidelity**
  grounds — validated on a real 1965 scan it matched/beat 0.17 on clean prose, was cleaner on marginal
  pages, and returned **empty (not hallucinated)** text on unreadable pages — but it is **~6-10x slower**
  (autoregressive) and reports a real per-block confidence (so the conf floor applies; the earlier
  "no confidence, VLM-like" assumption was wrong). A hard image can churn toward the token ceiling, so
  `Surya2Engine` caps `SURYA_MAX_TOKENS_FULL_PAGE` (`[ocr] surya2_max_tokens`, default 6144). `surya`
  (`surya_marker.SuryaEngine`) = local **0.17.x** — the fast, self-contained, **no-backend** `--engine
  surya` option (per-line confidence; the extra is pinned `<0.18`); its `identity()` stays `surya:0.17:*`
  (cache-stable — don't make it report the patch). Both are the same `surya-ocr` package at incompatible
  versions, so only one installs at a time (`[tool.uv].conflicts` locks them separately); each engine
  version-guards with a clear error.
- **Batched OCR (whole-book throughput).** `ocr_book` is 3-pass: build the page work-list + cache keys,
  `engine.run_batch` the cache-misses in chunks of `_OCR_BATCH` (32), then per-page write + degeneracy
  guard in spine order. `OCREngine.run_batch` defaults to sequential (`run` per page); `surya2`/`surya`
  override it to hand the whole chunk to one `RecognitionPredictor` call — Surya fans those out
  concurrently (`SURYA_INFERENCE_PARALLEL`, `[ocr] surya2_parallel`, default 8) and the vLLM server
  continuous-batches them. The win is for the served-VLM path (one HTTP round-trip per page was the wall);
  caching/`--force`/`--limit`/routing are unchanged. `run` is now just `run_batch([path])[0]`.
- **Windows console:** `cli.py` reconfigures stdout/stderr to UTF-8 (book text has em-dashes/ligatures
  that crash the cp1252 default). Keep CLI output ASCII-safe where practical (`->` not `→`).
- **PDF input** (`ingest_pdf.py`, opt-in `pdf` extra → **PyMuPDF, AGPL-3.0**): a page with a real text
  layer → `text` (preserved via `text_html`; `build_book` prefers it over re-OCR); an image page →
  `image`, **lossless-extracted** when it's a single full-page scan (best fidelity), else rendered at
  300 DPI. A rotated page is always rendered so rotation bakes into the pixels.
- **Slug collisions:** the project folder is slugged from the source **stem**, so `book.epub` and
  `book.pdf` share `book_projects/book/`, and ingest does **not** clear stale page images/OCR. Use
  distinct stems (or wipe the dir) when re-ingesting a different source under the same name.
- **`book_projects/`, `*.epub`, and `*.pdf` are gitignored** (large, regenerable); the `sample.epub`
  and `sample.pdf` fixtures are the tracked exceptions.

## Testing

There is **no automated test suite**. Verification is manual against the generated `sample.epub` and
`sample.pdf` fixtures (run `make_sample_epub.py` / `make_sample_pdf.py`, then `ingest`/`ocr`/`build`
them) and the `eval` harness against a hand-keyed gold JSON (`{page_index: ground_truth_text}`).
`scripts/bakeoff.py` compares
Surya raw/contrast/binarize + a VLM challenger on a page sample and writes only to `qa/` — never the
main `ocr/`. EPUBCheck (`validate.epubcheck`) needs a JRE + the 5.x jar via `$EPUBCHECK_JAR`.
