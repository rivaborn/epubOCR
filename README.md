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
- **OCR degeneracy guard — two tests, because one is not enough.** A repetition detector catches a
  single token looping; a *length outlier* test catches a model looping over **varied** filler, or
  fabricating on a blank page, which slips under the repetition threshold entirely. Measured: the
  length test caught 12 of 12 real failures where repetition caught 1, with no false positives over
  359 pages.
- **Cross-engine consensus** (`build --consensus <engine>`) — a second engine re-transcribes the
  least-trusted pages, and disagreement wins over a confident primary. This is the only guard that
  catches a **confident** fabrication: engines have been measured inventing whole pages at 0.95–0.99
  confidence, invisible to a confidence floor.
- **Multi-unit OCR** — fan one book across every free unit of a GPU fleet, loading the model onto an
  idle unit if none is serving it (see [`fleet.md`](fleet.md)).
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

## Configuring `config.local.toml`

`config.toml` is a committed template with `127.0.0.1` placeholders. Real endpoints go in
**`config.local.toml`**, which is **gitignored and therefore per-machine** — it does not travel with
a clone, so a new machine needs it written by hand. `config.py` prefers it over the template;
override the path with `$EPUBOCR_CONFIG`.

Target the **IP literal, not `localhost`** — an IPv4-only relay makes Python's happy-eyeballs stall
~5 s on IPv6 first.

### Minimum: a single served Surya 2

```toml
[ocr]
default_engine       = "surya2"
surya2_backend       = "vllm"
surya2_inference_url = "http://192.168.1.52:8000/v1"   # a server whose FIRST /v1/models entry is the model
surya2_model         = "surya-ocr-2"                   # must equal its --served-model-name
surya2_parallel      = 32                              # measured knee on both a 3090 and a GB10
```

⚠️ **Surya's attach compares only the FIRST entry of `/v1/models`.** A single-model server (a node's
own `host:port`) works; a multi-model gateway catalog does **not** — it fails with
`Model mismatch … got '<whatever is listed first>'`.

### Recommended with a fleet: let the pool resolve placement

If you run an [LLMConfig](https://github.com/rivaborn/LLMConfig) gateway, **do not name a lane.**
Which unit serves a model changes — on this lab `surya-ocr-2` moved twice in one day, silently
breaking a pinned URL. Ask the fleet instead:

```toml
[ocr]
default_engine      = "surya2"
surya2_backend      = "vllm"
pool                = "auto"          # or an explicit list: ["spark1", "spark2"]
pool_model          = "surya-ocr-2"   # served name to look for / request
pool_gateway        = "http://192.168.1.40:11430"
pool_autoload       = true            # load onto an IDLE unit when none is serving it
pool_autoload_units = 1               # a cold load costs minutes; a 2nd unit buys only ~1.55x
pool_max_units      = 4               # cap when several units already serve it
pool_chunk          = 32              # pages per unit per call — keep >= surya2_parallel
surya2_parallel     = 32
```

`pool = "auto"` uses units already serving the model, else loads it onto an idle one. It never takes
a unit that is mid-request, mid-swap, or under a non-preemptible lease. Set `pool_autoload = false`
to require a hand-loaded model.

> `pool_chunk` must be **>= `surya2_parallel`**, and `ocr_book` will raise the batch to
> `pool_chunk x units` automatically. If a pool ever reports all pages on one unit with the others
> at `0pg`, that ratio is why.

### Endpoints, roles, models (only for `--engine vlm` and `--llm` cleanup)

Three levels of indirection, because backends name the same model differently (Ollama tags
`qwen2.5vl:7b`; vLLM serves `qwen2.5-vl-7b`):

```toml
[endpoints.gateway]                                   # or [endpoints.ollama] / [endpoints.vllm]
base_url = "http://192.168.1.40:11430/v1"
api_key  = "EMPTY"                                    # Ollama ignores it, but the SDK requires one

[roles]                                               # role -> endpoint name
vlm_ocr      = "gateway"
text_cleanup = "gateway"

[models.gateway]                                      # alias -> that endpoint's model id
vlm_ocr      = "qwen2.5vl:7b"
vlm_ocr_hard = "qwen2.5-vl-32b"
text_cleanup = "qwen3:32b"
text_xhtml   = "qwen2.5-coder:32b"
```

Leave these out entirely if you only run Surya/Tesseract — the default build never calls an endpoint.
Check what resolved with `uv run epubocr show-config` and `uv run epubocr endpoints`.

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
