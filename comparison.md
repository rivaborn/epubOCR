# OCR improvement comparison — *This Town* (faded scan)

Goal: improve the OCR output for `This Town.epub`, a 536-page image-only EPUB that turned out to
be a **badly faded, low-contrast scan**. This documents the preprocessing / engine bake-off, the
cross-engine consensus check, and the resulting build.

## TL;DR

- **Raw Surya is the best version.** External preprocessing did **not** help — Surya's own
  normalization already handles the faded scan better than added contrast or binarization.
- The scan is genuinely degraded: mean OCR confidence ≈ **0.49**; ~88% of pages are below 0.60.
- The fidelity-correct output is therefore a **searchable facsimile EPUB**: **7 pages reflowable,
  529 facsimile** (page image + hidden OCR text layer), not a reflowable book.
- The only lever that would materially help is a **better source scan**.

## Baseline (raw Surya, all 536 pages)

| Metric                         | Value           |
| ------------------------------ | --------------- |
| Mean OCR confidence            | **0.488**       |
| Pages ≥ 0.80 (reflowable)      | 9 (1%)          |
| Pages 0.60–0.79                | 54              |
| Pages < 0.60 (→ facsimile)     | 466 (88%)       |

## Preprocessing bake-off (Surya on a page sample)

Two passes. The first used an aggressive contrast (`autocontrast cutoff=1, 1.6×, +deskew`); the
second a gentle one (`1.3×, no deskew`) after the aggressive setting was found to destroy pages.

| Variant            | mean_conf | pages ≥0.80 | Notes                                                              |
| ------------------ | --------- | ----------- | ----------------------------------------------------------------- |
| **surya-raw**      | **0.55**  | 2 / 12      | Most robust — produces text on every page. **Winner.**            |
| surya-contrast     | 0.51      | 2 / 12      | Worse than raw on most pages once it stops failing                |
| surya-binarize     | 0.43      | 0 / 12      | Adaptive threshold + CLAHE adds speckle on faded scans — worst    |

**The aggressive-contrast trap:** in the first pass contrast *appeared* best (0.58 vs raw 0.50) —
but only because it **wiped the text entirely (`n/a`) on 9 of 24 pages**, and those failures dropped
out of the mean. With a gentle contrast that doesn't fail, raw beats it (0.55 vs 0.51). Lesson:
score preprocessing on *all* pages, counting wipe-outs as failures, not just where it succeeds.

## Cross-engine consensus (Surya ↔ Qwen2.5-VL-7B)

Where two independent engines agree, the text is trustworthy. On this scan they **mostly don't**:

| Metric                                   | Value     |
| ---------------------------------------- | --------- |
| Mean Surya↔VLM agreement (12 pages)      | **0.27**  |
| Agreement on a confident page (p446)     | 0.75      |
| VLM degenerate (repetition loop) pages   | 1 / 12    |

Low agreement on faded pages = neither engine is reliable there, which is exactly the signal to send
those pages to facsimile. Agreement rises only where Surya is already confident.

## The 32B VLM (`qwen2.5-vl-32b`)

Evaluated as a stronger challenger but **impractical to sweep**: on the single RTX 3090 it runs with
CPU-offload (eager) at roughly **3–4 min/page**, and it can't co-reside with a second model. It also
*confabulates* on faded text (invents plausible names not on the page) — a bigger model fails more
convincingly, with no confidence signal. Used only as a spot-check; not a fix.

## Infrastructure change adopted: LLMConfig gateway

Mid-investigation the box gained **LLMConfig** (`:11430/v1`), a GPU arbiter with an OpenAI-compatible
gateway that **auto-loads the requested model** (evict-wait then load) and adds a second 8 GB
companion lane. This removed the contention that had blocked the VLM (7B couldn't load while 32B held
the 3090). The pipeline now points VLM/LLM at the gateway (`config.local.toml`), so model selection is
hands-off.

## Build fix: confidence-based routing

The original build routed on "has any OCR text," so it labeled **527 garbled low-confidence pages as
reflowable**. Fixed: pages below the confidence floor (0.80) now go to **facsimile with a hidden,
searchable OCR text layer**. VLM pages (no confidence) stay reflowable and rely on the degeneracy
guard + fidelity verifier instead.

| Build                         | reflowable | facsimile | meaning                                  |
| ----------------------------- | ---------- | --------- | ---------------------------------------- |
| Baseline (buggy routing)      | 527        | 9         | garbled text mislabeled as clean         |
| **Fixed (confidence routed)** | **7**      | **529**   | honest: clean where readable, else image |

## Result & recommendation

The improved EPUB (`book_projects/This_Town/output/improved.epub`) is a **searchable facsimile**:
7 reflowable pages where Surya is confident, 529 facsimile pages (original image + searchable OCR
layer) with a `page-list` nav. For a scan this faded that is the correct, fidelity-preserving output —
reflowing low-confidence OCR would fill the book with garbled or confabulated text.

**To do materially better, get a cleaner source scan.** No preprocessing, engine, or larger VLM
recovers faded ink that isn't legibly present; they only risk confident hallucination.
