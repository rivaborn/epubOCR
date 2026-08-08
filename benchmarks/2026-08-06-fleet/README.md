# Fleet OCR bake-off — 2026-08-06

The raw data behind [`fleet.md`](../../fleet.md) §10–§13. Kept because every number in that
document is only as good as the run that produced it, and a claim you cannot re-derive is a
claim you eventually stop trusting.

## Files

| File | What it is |
| ---- | ---------- |
| `parallel_sweep.py` / `.json` | the `surya2_parallel` sweep on spark3 (§12) — 8→128 over a fixed 128-page sample |
| `eval_fleet.py` / `fleet_eval_results.json` | the first 4-engine accuracy probe (§10) — surya2 / chandra / paddleocr / surya 0.17 |
| `score_runs.py` / `score_summary.json` | whole-book scoring (§11): gold CER, distribution, cross-engine agreement, over-production |
| `chandra_sample.py` / `chandra_sample.json` | Chandra on a designed 29-page sample (§11) — gold pages + the 12 pages PaddleOCR corrupted |

The gold set itself lives at [`tests/fixtures/mustee.gold.json`](../../tests/fixtures/mustee.gold.json)
— 4 pages of *The Mustee* transcribed **by hand from the page images**, as printed
(original hyphenation, curly quotes, the blank page recorded as `""`). It is the only
absolute accuracy reference in this repo besides `sample.gold.json`, and it cost real
effort to make; re-key it rather than guessing if it is ever lost.

## Running them again

Each script hardcodes the endpoints it measured against (`192.168.1.5x:800x`, spark3's
slot ports). **Those are a snapshot** — see `fleet.md` §13: placement on this fleet moved
twice in the day these were written. Re-point them at whatever is serving the model before
believing a re-run.

⚠️ `score_runs.py` reads per-engine OCR output from sibling directories
(`ocr_surya2/`, `ocr_paddleocr/`, `ocr_chandra/`, `ocr_baseline_surya017/`) that were
scratch copies of `book_projects/<slug>/ocr/`. They are **not** committed — the page text
is regenerable from the content-addressed cache, and `book_projects/` is gitignored for
good reason (large, regenerable). The scored *results* are what is preserved here.

## The two findings worth re-reading before trusting any of it

1. **Speed and fidelity disagreed, and fidelity won.** PaddleOCR-VL was 3x the fastest
   engine measured and is disqualified: it silently corrupted 12 of 359 pages at confidence
   0.953–0.984 against a book mean of 0.981 — invisible to both the confidence floor and
   the repetition guard.
2. **A per-unit rate is not a unit's speed.** With a generative OCR model, time tracks
   *output tokens*. One pathological page (a 12k-char fabrication) made one Spark look 4.7x
   slower than an identically configured sibling, and it took a second look to notice both
   nodes held the same models.
