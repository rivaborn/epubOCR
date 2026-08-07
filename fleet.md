# OCR on the fleet — optimal engine, placement, and process for epubocr

`epubocr` was designed for "a laptop plus an optional GPU host" ([epubOCR.md](epubOCR.md)
§Topology). The homelab is now a six-unit fleet behind the LLMConfig gateway
(`192.168.1.40:11430`): an RTX 3090, an RTX 3070 Ti slot lane, and four DGX Spark GB10
nodes. This doc decides, per pipeline stage, the optimal model and placement — and proposes
an "OCR mode" fleet arrangement. It is analysis only: the recommended code/config changes
are listed as future work (§9), not made here. Numbers are labeled **[measured — provenance]**
or **[predicted — bake-off #N]** (§10).

## TL;DR — the calls

> ⚠️ **§1–§9 are the analysis as written BEFORE anything was measured. §10–§12 measured it,
> and three of the calls below did not survive.** Read the bullets as the reasoning, not the
> conclusions:
>
> * **"a GB10 only maybe wins at batch >=128, which the client can't currently drive"** —
>   wrong twice. The knee is **32**, the client drives it today, and a GB10 does 2.3 s/page
>   (§12). The bandwidth arithmetic in §4 under-modelled prefill and over-modelled decode.
> * **"PaddleOCR-VL is the bulk winner"** (implied by its 0.60 s/page in §11) — it is
>   **disqualified on fidelity**: it silently corrupts 3.3 % of pages at normal confidence,
>   under both existing guards (§11).
> * **"Chandra is the strongest challenger candidate"** (§4) — it fabricates on unreadable
>   pages at conf 0.994 (§11), so it cannot be trusted where a challenger is most needed.
>
> What did survive: surya2 keeps the ground-truth slot; consensus is worth wiring in (it is,
> now); and Sparks should host surya2 **co-resident**, not dedicated (§12).

- **Bulk OCR: borrow the 3090.** `surya2` served from a 3090 vLLM slot is the measured
  5.23x batched path **[measured — commit `06cb7eb`, 2026-08-01]** with a measured
  concurrency sweet spot of 32 **[measured — `config.toml:54`]**. The companion (3070 Ti)
  slot stays the always-resident *ambient* engine for incremental/small jobs. A Spark-hosted
  surya2 recipe is a **bake-off candidate, not a recommendation** — the bandwidth arithmetic
  (§4) says a GB10 loses ~3.4x at today's `parallel=32` and only maybe wins at batch ≥128,
  which the client can't currently drive.
- **VLM challenger: phase it, don't co-host it.** Run the consensus/hard-page VLM
  (`qwen2.5-vl-32b`) on the 3090 *after* bulk OCR completes — the 3090 flips back to vl32
  anyway. A Spark-hosted VLM is the future "both at once" upgrade; **no chat VLM exists on
  any Spark today** (verified against the gateway catalog, 2026-08-06).
- **LLM cleanup: DeepSeek-V4-Flash on spark1+2** — already resident (zero fleet churn),
  MoE + continuous batching is the one workload shape a GB10 is genuinely good at, and it
  takes cleanup off the 3090 entirely. But it is **wasted until `clean_page` is
  parallelized**: the cleanup loop is strictly serial, one blocking HTTP call per page
  (`pipeline.py:304-318`), with no timeout and unbounded `max_tokens`.
- **The fidelity-first win is free: wire `consensus.assess` into `build_book`.** The
  cross-engine agreement signal already proved itself on *This Town*
  ([comparison.md](comparison.md) §Cross-engine consensus) but `consensus.py` is dead code
  in the pipeline — only `scripts/bakeoff.py` calls it.
- Operationally, all of this is one LLMConfig **cookbook state** away (§8): only the 3090
  changes role, twice, and every Spark stays exactly as the daily driver left it.

## 1. The fleet, as epubocr sees it

Verified live 2026-08-06 against `GET /api/status` and the gateway `/v1/models` catalog.

| Unit        | Hardware               | Resident today                                      | epubocr's use today                     |
| ----------- | ---------------------- | --------------------------------------------------- | --------------------------------------- |
| `primary`   | RTX 3090 24 GB         | **free** (usually `qwen2.5-vl-32b`, "vl32", pinned) | `vlm_ocr_hard` challenger, via gateway  |
| `companion` | RTX 3070 Ti 8 GB       | `surya-ocr-2` (util 0.54, ctx 12288) + 1.5b relay   | **default OCR engine** (`surya2`)       |
| `spark1+2`  | 2x DGX Spark (tp pair) | `deepseek-v4-flash` tp=2, ctx 262144                | none                                    |
| `spark3`    | DGX Spark GB10         | `qwen35-122b` (Intel int4 AutoRound), ctx 262144    | none                                    |
| `spark4`    | DGX Spark GB10         | `qwen3-vl-reranker-8b` + `qwen3-vl-embedding-8b`    | none                                    |

Two structural facts drive every placement decision below:

1. **GB10 shape.** Each Spark has 128 GB unified memory but ~273 GB/s of bandwidth — versus
   the 3090's 24 GB at ~936 GB/s. A GB10 is a *continuous-batching / MoE machine*: it wins
   when many concurrent requests share one weights-read per decode step, and loses any
   single-stream dense race by roughly the bandwidth ratio (~3.4x).
2. **LLMConfig affordances.** The gateway auto-loads on first request; `/lane/<unit>/v1`
   path-pinning is how header-less clients (Surya's attach) reach a specific unit — and it
   is the **only** supported remote route to a slot: exposing the per-slot relays to the LAN
   was tried and does not work under WSL mirrored networking, so the "direct relay `:11438`"
   fallback sketched in `config.toml:62-64`'s comment block is a dead end and should be
   deleted. Cookbook states snapshot/restore the whole fleet; `X-LLM-Hold` gives a client an
   auto-renewing lease; cold loads cost **≥250 s** on a Spark and ~99 s on the 3090
   **[measured — LLMConfig wiki, 2026-07-26]**.

The companion slot serves surya2 with **KV cache deliberately kept bf16** because epubocr
reads per-block logprobs for `mean_conf` — any future re-hosting of surya2 must preserve
that, or the confidence floor (§7 of [epubOCR.md](epubOCR.md)) goes blind.

## 2. Where the pipeline actually spends time

The pipeline's fan-out story determines which units can help at all:

- **surya2 is the only stage that fans out.** `ocr_book` chunks cache-misses into batches of
  `_OCR_BATCH = 32` (`pipeline.py:30`) and hands each chunk to `engine.run_batch`, which
  makes **one** `RecognitionPredictor` call; Surya fans that out as `SURYA_INFERENCE_PARALLEL`
  concurrent HTTP requests (`ocr/surya2.py`, `surya2_parallel = 32`). But the target is a
  **single global URL** — Surya's process-wide settings singleton holds one
  `SURYA_INFERENCE_URL`, so there is exactly one destination and one concurrency knob.
  Constraint from commit `c673248`: whatever lane serves it must list `surya-ocr-2` **first**
  in its `/v1/models`, which is precisely what the lane-scoped gateway path guarantees.
- **`--llm` cleanup is serial.** `_cleaned_body` → `clean_page` runs once per reflowable
  page, inside the page loop, one blocking call, no thread pool, no timeout, no
  `max_tokens` cap (`pipeline.py:304-318`, `llm/cleanup.py:57-83`). A Spark's continuous
  batching gives this exactly nothing today.
- **The `vlm` engine has no `run_batch` override** (`ocr/vlm_openai.py`) — one sequential
  round-trip per page, `conf=None` always.
- **The eval harness is fidelity-only.** `eval/harness.py` calls `ocr_image` one page at a
  time, so it can gate *what* an arrangement transcribes but says nothing about throughput.
  Throughput claims need `scripts/bakeoff.py` (or a timed `ocr --force --limit N` run).

So: the fleet's width is exploitable **where the code already fans out** (surya2's 32
concurrent requests) and **where fan-out is cheap to add** (cleanup, §6). Nothing else
parallelizes without new code.

## 3. Bulk OCR — engine and placement

Placement changes **none of the fidelity properties**: same model, same logprobs → same
`mean_conf`, same degeneracy guard. This is purely a throughput decision — which is why the
eval harness can't arbitrate it and a throughput bake-off must.

The arithmetic (all **[predicted — bake-off #1]** unless labeled): `surya-ocr-2` is ~3 GB
bf16 (~1.5 B params). A batch of concurrent decodes reads the weights once per step, so the
step-rate ceiling is bandwidth-bound: **3090 ≈ 936/3 ≈ 310 steps/s; GB10 ≈ 273/3 ≈ 91
steps/s** — the 3090 is ~3.4x faster at equal batch. Aggregate tokens/s scales with batch
until compute-bound: at ~3 GFLOP/token, GB10 × batch 128 ≈ 11.6k tok/s ≈ 35 TFLOPS, inside
a GB10's bf16 envelope — so a Spark *could* pass the 3090-at-batch-32 (~9.9k tok/s) somewhere
around batch ≥128. But OCR requests are also vision-prefill-heavy, and the measured 3090
plateau (24–48, 64 regresses) shows the simple decode model isn't the whole story.

| Option                          | Verdict                                                                                                                                                                                                                                                                                                              |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **companion slot** (current)    | Keep as the **ambient default** — always resident, zero setup, correct for a chapter or a re-run. But its 0.54-util slice of 8 GB caps concurrent KV slots well below `parallel=32`, so requests queue server-side on a big book. Its real ceiling is bake-off #2.                                                     |
| **3090 vLLM slot**              | **Bulk winner.** 5.23x batched **[measured — commit `06cb7eb`]**; parallel sweet spot 32, plateau 24–48, 64 regresses **[measured — `config.toml:49`]**; the card is even free right now. Cost: vl32 is displaced for the duration (restored in Phase B, §8).                                                          |
| **single Spark** (new recipe)   | **Bake-off candidate #1, not a recommendation.** ~3.4x slower at batch 32; crossover *predicted* somewhere ≥128 concurrent — beyond anything measured (`surya2_parallel` has never been swept past 64, and the 64-regression was observed on the 24 GB card, so it may not transfer). Small model → staging is minutes, not hours; the cost is a new sparkrun recipe + catalog row. |
| **4 Sparks sharded**            | **Blocked by the single global URL** in Surya's settings singleton. Workarounds: a gateway-side round-robin of one served name across lanes (doesn't exist — bake-off #8 asks LLMConfig), or N processes each with its own `EPUBOCR_CONFIG` pointing at a different `/lane/sparkN/v1` and a `--limit`/page-range shard (works today, ugly). Archive-scale only; one 3090 already saturates a per-book run. |

**Call: 3090 for bulk, companion for ambient.** Revisit only if bake-off #1 shows a GB10
crossing over at high batch — which would matter for multi-book archive runs, not single
books.

## 4. The VLM — hard pages and the challenger role

Config today: `vlm_ocr = qwen2.5vl:7b`, `vlm_ocr_hard = qwen2.5-vl-32b`
(`config.local.toml:36-37`), both via the gateway.

- The damning "3–4 min/page, impractical to sweep" figure for the 32B
  ([comparison.md](comparison.md) §The 32B VLM) is from the **CPU-offload era**, before
  LLMConfig owned the card. On a dedicated, fully-resident 3090 it should be seconds per
  page **[predicted — bake-off #3]** — likely flipping vl32 from "spot-check only" to
  "viable challenger over the whole low-confidence population."
- The fidelity policy is unchanged by any of this: the VLM stays a *challenger*, never the
  transcription source ([epubOCR.md](epubOCR.md) §OCR engines). Bigger VLMs fail more
  convincingly — the 32B *confabulated names* on faded pages **[measured — comparison.md,
  This Town]** — so its output feeds `consensus.assess`, not the book.
- **A GB10 is actually a good challenger host**: hard-page traffic is a small,
  latency-tolerant sample; vision prefill is compute-heavy (the GB10's relative strength)
  and decode is short. But no Spark VLM recipe exists, and one would need a
  sparkrun-compatible build (bake-off #7 — ideally a Qwen3-VL MoE) plus weights staging.
  spark4 is the natural host (its pooling models leave headroom).
- **The strongest challenger candidate is not a general VLM at all: Chandra-OCR-2**
  (datalab's flagship, same vendor as surya). **85.9% olmOCR-bench vs surya2's 83.3%**
  [reported — datalab, 2026], structure-preserving output (tables/handwriting/forms),
  vLLM-served, quoted at 1.44 pages/s on an H100 at 96 concurrent — the wide-concurrency
  shape a GB10 handles well. A purpose-trained OCR model is a better second opinion than
  qwen2.5-vl-32b, and the consensus role needs no confidence signal, just an independent
  transcription. Caveats: no documented confidence output (so it cannot *replace* surya2
  as ground truth without logprob→conf engineering), weights are modified OpenRAIL-M
  (fine for personal use), and its behavior on unreadable pages — surya2's decisive
  virtue — is unmeasured here (bake-off #9).

**Call: near-term, phase the work** — bulk surya2 on the 3090 (Phase A), then restore vl32
and run the consensus pass (Phase B). The Spark VLM is the future upgrade that lets both
run concurrently and permanently un-pins the challenger from the 3090.

## 5. LLM cleanup — model choice, and the serial-bottleneck reality

**The model choice is irrelevant until the serial loop is fixed.** Any of these models,
serially, is hours per book; any of them behind a 16-way pool is minutes:

| Candidate                      | Where     | Single-stream (serial loop)                                                                       | With a 16-way pool                                                     |
| ------------------------------ | --------- | ------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| **`deepseek-v4-flash`**        | spark1+2  | ~600-token page at MoE decode rates ≈ 15–30 s/page → **2–4.5 h** for a 536-page book [predicted]  | **~10–20 min** [predicted — bake-off #4]; continuous batching is what a GB10 pair is *for* |
| `qwen35-122b` (int4, A10B)     | spark3    | similar order; ~5 GB active-params/step at 273 GB/s ≈ 55 steps/s ceiling [predicted]              | similar; the **quality challenger** for the cleanup bake-off           |
| `qwen2.5-coder:32b`            | 3090      | fast decode but **contends with OCR/VLM for the card**                                            | fallback when the Sparks are down                                      |

**Call: DS4-Flash on spark1+2.** It is already resident (zero cookbook churn), its 262k
window shrugs at any page, MoE-under-concurrency is the GB10's best event, and it keeps the
3090 wholly owned by OCR (Phase A) or the challenger (Phase B). The fidelity verifier
(`llm/fidelity_verifier.py`) is the safety net either way — a *stronger* cleanup model
mostly buys a lower hold rate, which is exactly what bake-off #4 measures.

**Bug to fix while touching this** (§9): `clean_page` builds its client for role
`text_cleanup` but resolves model alias **`text_xhtml`** (`llm/cleanup.py:67-68`) — so the
pipeline actually calls `qwen2.5-coder:32b` while `show-config` reports the `text_cleanup`
alias `qwen3:32b`. Whichever is intended, make the aliases agree and make `show-config`
print the truth.

## 6. Consensus — wire the dead code

`consensus.agreement`/`assess` exist, are tuned (`agree_floor=0.85`, `conf_floor=0.80`),
and are exercised only by `scripts/bakeoff.py`. `build_book` never calls them — yet the
signal is proven: agreement 0.27 on faded pages vs 0.75 on a confident one **[measured —
comparison.md, This Town]**, i.e. it separates exactly the pages the confidence floor
routes to facsimile.

**Recommendation:** in `build_book`, for pages with `mean_conf < conf_floor` (and
optionally a small random canary sample of confident pages), fetch a VLM transcription and
run `assess`:

- **agreement ≥ floor** → promote the page to reflowable (or at minimum annotate it
  trusted) — two independent engines agreeing beats one engine's self-report;
- **disagreement** → facsimile confirmed, now with evidence instead of a single scalar.

Cost is bounded by the low-confidence population: a handful of pages on a good scan, most
of the book on a *This Town* — so cap it with a budget flag (`--consensus-max-pages N`).
This is the stated design intent of [epubOCR.md](epubOCR.md) §OCR engines finally landing
in the router, and it is the single highest-fidelity-per-effort change in this doc.

## 7. "OCR mode" — the cookbook state

The operational proposal. Only the 3090 changes state — that is the selling point: no
Spark unload (whose VRAM clears *after* residency — wait for the VRAM figure), no
`needs_empty_node` shuffle, and the daily driver survives untouched.

| Unit        | Normal (daily driver)        | Phase A — bulk OCR                  | Phase B — consensus + build            |
| ----------- | ---------------------------- | ----------------------------------- | -------------------------------------- |
| `primary`   | vl32 (pinned)                | **surya2 vLLM slot** (5.23x path)   | vl32 restored — consensus challenger   |
| `companion` | surya-ocr-2 + relay (pinned) | unchanged (ambient OCR stays live)  | unchanged                              |
| `spark1+2`  | DS4-Flash tp=2               | unchanged — **cleanup engine**      | unchanged — cleanup continues          |
| `spark3`    | qwen35-122b                  | unchanged (idle / cleanup QA judge) | unchanged                              |
| `spark4`    | reranker + embedder          | unchanged; future VLM-recipe host   | unchanged                              |

Transition mechanics:

- Save the daily driver as a cookbook state (it already exists as `DF4_RAG Daily Driver`),
  save an `OCR Mode` state, and `apply` to flip. Each 3090 flip is a cold load — ~99 s to
  ~250 s **[measured — LLMConfig wiki]** — and there are **two per run** (A→B). Amortized
  over a 500-page book this is noise; for a 30-page pamphlet just use the companion slot.
- Hold an **`X-LLM-Hold`** (or an explicit lease) on the 3090 surya2 slot for the duration
  of Phase A so auto-placement can't evict it mid-chunk.
- Expected per-stage throughput, honestly labeled:

| Stage                       | Rate                                                                       |
| --------------------------- | -------------------------------------------------------------------------- |
| bulk OCR, 3090              | 5.23x the pre-batch path **[measured — `06cb7eb`]**; absolute pages/min = bake-off #1 |
| bulk OCR, companion         | baseline; real ceiling = bake-off #2                                       |
| cleanup, DS4 pooled         | ~10–20 min/book **[predicted — bake-off #4]**                              |
| consensus pass, vl32        | seconds/page **[predicted — bake-off #3]**                                 |

## 8. Chapter-flow work is orthogonal

The uncommitted work in the tree (sections, outline-driven chapter grouping, paragraph
stitching in `assemble.py`/`ingest*.py`/`pipeline.py`) neither touches nor is touched by
any of this — it consumes OCR text after routing. No interaction to plan around.

## 9. Future config / CLI / code deltas (recommendations only)

| # | Change                                                                                                              | Where                          |
| - | ------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| 1 | Expose `--conf-floor` on `build` (param exists at `pipeline.py:193`, CLI never surfaces it)                          | `cli.py`                       |
| 2 | Fix the `text_cleanup`/`text_xhtml` alias mismatch; make `show-config` print the model actually used                 | `llm/cleanup.py:67-68`, `cli.py` |
| 3 | Delete the dead-end "direct relay `:11438`" comment (LAN exposure doesn't work under WSL mirrored networking)        | `config.toml:62-64`            |
| 4 | Bounded thread pool for cleanup (`cleanup_parallel`, 8–16) + `cleanup_max_tokens` + `cleanup_timeout`, timeout → deterministic fallback | `pipeline.py`, `llm/client.py` |
| 5 | Wire `consensus.assess` into `build_book` for low-conf pages, with `--consensus-max-pages`                           | `pipeline.py`                  |
| 6 | Multi-URL surya2 (list-valued `surya2_inference_url`) *or* document the `EPUBOCR_CONFIG`-per-shard pattern           | `ocr/surya2.py` / docs         |
| 7 | Throughput mode for `bakeoff.py` (timed `run_batch` sweeps); the eval harness stays fidelity-only, by design         | `scripts/bakeoff.py`           |

## 10. Measured 2026-08-06 — first accuracy bake-off (answers #9, part of #1/#3)

Fleet was rearranged same day: surya-ocr-2 + chandra-ocr-2 co-resident on **spark3**
(ports 8000/8001, direct single-model `/v1/models` — satisfies surya's attach), and
**paddleocr-vl-1.6** (0.9B, `PaddlePaddle/PaddleOCR-VL-1.6`) on both GPU lanes. Scored
against a hand-transcribed gold set from The_Mustee (real 1965 scan): 3 clean prose
pages + 1 blank page as hallucination probe. CER-folded = curly quotes/dashes
normalized; chandra additionally HTML-stripped (it emits `<div data-bbox>` layout
markup regardless of prompt — its native format, ~0.20 raw CER of pure markup).

| Engine (single-stream)  | CER (prose, folded)  | Blank page          | conf?  | s/page     |
| ----------------------- | -------------------- | ------------------- | ------ | ---------- |
| surya2 @ spark3         | 0.0004–0.0048        | empty ✓             | 0.99   | 21–32      |
| chandra @ spark3        | 0.0034–0.0071*       | empty ✓             | none   | 23–48      |
| paddleocr-vl @ 3070 Ti  | 0.0044–0.0055        | whitespace only ✓   | none   | 4.6–8.5    |
| surya 0.17 (cached)     | 0.0000–0.0029        | **hallucinated**, conf 0.42 (floor catches it) | 0.99 | — |

\* HTML-stripped; also note chandra de-hyphenates line breaks itself (costs it vs
as-printed gold, but is *cleaner* input for downstream — `layout.py` does this anyway).

Takeaways: **on clean scans all four are within ~0.7 % CER of each other** — accuracy
does not separate them there; the discriminators are the confidence signal (surya only),
blank-page behavior (surya 0.17 alone hallucinated), and speed. **PaddleOCR-VL is the
surprise: surya2-class accuracy at 4–5x the single-stream speed on the smallest GPU** —
a serious bulk-OCR candidate, needing only a confidence story (vLLM logprobs) and a
faded-scan validation. surya2 single-stream on a GB10 is 21–32 s/page, confirming §3's
bandwidth arithmetic — its Spark placement only pays at high `surya2_parallel`. The
open accuracy question is now the **faded-scan case** (This Town-class input), where
none of the challengers have been probed for confabulation.

## 11. Measured 2026-08-06 (later) — whole-book, multi-unit

Fleet placed as: surya2 on **spark1+spark2**, chandra on **spark2+spark3**, PaddleOCR-VL
on the **3090**. Book: *The Mustee* PDF, **359 image pages**, a clean 1968 hardback scan.
Every run `--force` (no cache). This is the first test of the new
[`ocr/pool.py`](epubocr/ocr/pool.py) multi-unit fan-out.

### Throughput

| Engine       | Units      | Wall   | s/page | Note                                  |
| ------------ | ---------- | ------ | ------ | ------------------------------------- |
| paddleocr-vl | 1x 3090    | 214 s  | 0.60   | fastest by 3x                         |
| surya2       | 2x GB10    | 668 s  | 1.86   | split 199/160; **1.55x** over 1 unit  |
| surya2       | 1x GB10    | 1034 s | 2.88   | accidental baseline (pool bug, below) |
| chandra      | 2x GB10    | —      | 11.45  | 29-page designed sample               |

🔴 **The headline throughput fact: concurrency, not placement, is what matters.** Surya2
single-stream on a GB10 measured **21–32 s/page** (§10); at `surya2_parallel = 32` the
*same* model on the *same* node is **2.88 s/page** — ~10x. §3's arithmetic predicted the
Spark could only win at high batch and treated that as unreachable; it is reachable, and
the Spark is a good surya2 host. Pool scaling is real but sublinear: **1.55x for 2x the
hardware**.

> ⚠️ **A per-unit rate is not a unit's speed.** The chandra run showed spark2 at
> 41.5 s/page against spark3's 8.8 s/page, which reads as a co-residency penalty and is
> **not** one — both nodes held exactly the same two models. spark2's chunk simply
> contained the blank page chandra fabricated 12,147 chars on: 25,396 output chars across
> its 8 pages versus a 2,171-char average elsewhere. Generation time tracks *output
> tokens*, so with a generative OCR model a fixed-size chunk is not a fixed-size unit of
> work, and the pool's dynamic stealing cannot rebalance a straggler already in flight.
> Read per-unit rates only alongside output volume.

### Accuracy, and the finding that decides it

| Engine    | gold CER (n=4) | prose CER | blank page | over-produced pages | agreement w/ surya2 |
| --------- | -------------- | --------- | ---------- | ------------------- | ------------------- |
| surya2    | **0.0018**     | 0.0004–0.0048 | **empty ✓** | **0 / 359**      | —                   |
| paddleocr | 0.0035         | 0.0044–0.0048 | whitespace ✓ | **12 / 359 (3.3%)** | 0.969 mean       |
| chandra   | 0.2526         | 0.0013–0.0056 | **12,147 chars fabricated** | 0 / 12 tested | 0.999 mean |
| surya 0.17| 0.2507         | 0.0000–0.0029 | 19 chars hallucinated | 0 / 359 | 0.991 mean |

On **clean prose all four are within ~0.5 % CER** — accuracy does not separate them
there, and chandra tracks surya2 at 0.999 agreement across every ordinary page tested.
They separate completely on **pages with nothing to read**:

🔴 **PaddleOCR-VL silently degenerates on 3.3 % of pages.** Twelve pages ran to the token
cap emitting instruction-loop filler — *"Input the name of the hospital."*, *"The text is
not printed."* — repeated to 4.6–9.1 k chars against a ~2.2 k-char norm. It does this at
**confidence 0.953–0.984 against a book mean of 0.981**, and at repetition ratios of
0.03–0.33, i.e. **under the 0.35 degeneracy threshold on 11 of the 12**. Both existing
guards miss it. Chandra reproduced *none* of these (0.995–0.999 agreement with surya2 on
the same twelve), so the pages are ordinary prose — **the failure is paddle-specific,
not page-specific.**

🔴 **Chandra fabricates on blank pages, at maximum confidence.** On the book's one blank
page it emitted **12,147 characters** of invented JSON (a synthetic run of years
1842→2004 with made-up coordinates), `conf = 0.994`, repetition ratio 0.101 — again under
the guard. On the cover it invented *"OFF-SCREEN MAGIC / by LANCE HONNELL"*, and produced
**different** fabrications on repeated runs of the same image. surya 0.17 hallucinates on
the same blank page but its confidence collapses to 0.42, so the conf floor catches it;
chandra's does not move.

**Conclusion: surya2 keeps the ground-truth slot, and it is not close.** It is the only
engine of the four that returned *empty* rather than invented text on unreadable input,
the only one with zero over-produced pages across 359, and it has the best gold CER. Its
cost is 3x paddle's wall clock — cheap insurance against a 3.3 % silent-corruption rate
that no guard in the pipeline currently detects.

**Two concrete consequences for the code:**

1. **Wire `consensus.assess` into `build_book`** (§7 was a recommendation; it is now
   evidence-backed). Cross-engine disagreement is the *only* signal that caught paddle's
   12 pages — confidence and repetition both failed. A cheap version already works: a
   page whose length exceeds ~2x a second engine's is the whole signature.
2. **The degeneracy guard needs a second test.** `repetition_ratio` keys on the single
   most frequent *token*, which a loop of varied phrases evades (0.03–0.33 on pages that
   are unambiguously degenerate). A length-vs-book-median outlier test would have caught
   all 12 paddle pages and chandra's blank page, at no model cost.

### Pool bugs found by running it (both fixed)

* **A silently single-unit pool.** `ocr_book` batches to `_OCR_BATCH = 32` and the pool
  chunk was also 32, so every call held exactly one chunk: the first worker took it and
  every other worker returned immediately. All 359 pages went to spark1 with **no error**
  — the only tell was `spark2 util=0%`. Fixed with `PoolEngine.preferred_batch`
  (chunk x units), which `ocr_book` now honours.
* **`usage=active` blocked an explicitly named unit.** A unit reads `active` for 60 s
  after its own last request, so naming a unit you *just* used failed. Discovery
  (`pool = "auto"`) still skips busy units — that is the point — but an explicit list now
  only requires that the unit serve the model.

## 12. Measured 2026-08-06 (later still) — the `surya2_parallel` sweep (answers #1/#2)

§11 left the obvious question open: if concurrency is worth ~10x and a second *unit* only
1.55x, **how far does concurrency go?** Swept on **spark3 with `surya-ocr-2` as its only
resident model**, so nothing else on the node contends.

Method (it decides whether the numbers mean anything): the same 128 pages in the same
order for every setting, evenly spaced across the book, decoded once up front and excluded
from the timing; one engine instance re-tuned between runs (the backend reads
`SURYA_INFERENCE_PARALLEL` at `generate()` time, so no setting pays a reconnect the others
do not); a warm-up batch before the first measured run, so `parallel=8` is not charged for
warming the server and made to look artificially bad — which would manufacture a scaling
curve out of nothing. Sample >= max parallel, or 128-way concurrency could not be exercised.

### The Mustee (clean 1968 scan), 128 pages

| parallel | wall  | s/page | pages/s | vs p8  | marginal |
| -------- | ----- | ------ | ------- | ------ | -------- |
| 8        | 583.4 | 4.558  | 0.219   | 1.00x  | —        |
| 16       | 433.4 | 3.386  | 0.295   | 1.35x  | 1.35x    |
| 24       | 340.7 | 2.661  | 0.376   | 1.71x  | 1.27x    |
| **32**   | 309.1 | 2.415  | 0.414   | 1.89x  | 1.11x    |
| 48       | 297.0 | 2.320  | 0.431   | 1.96x  | 1.04x    |
| 64       | 296.7 | 2.318  | 0.431   | 1.97x  | 1.00x    |
| 96       | 298.9 | 2.335  | 0.428   | 1.95x  | 0.99x    |
| 128      | 294.7 | 2.303  | 0.434   | 1.98x  | 1.01x    |

🔴 **It does not keep scaling. The knee is 32 and everything from 48 up is one flat line**
(48/64/96/128 span 294.7–298.9 s — a **1.4 %** spread, i.e. noise). Total available gain
over `parallel=8` is **1.98x**, of which `parallel=32` already captures **96 %**. The
current `[ocr] surya2_parallel = 32` is correctly placed; 48 is defensible for ~4 %, and
anything above that is measuring the weather.

Two secondary findings:

* **The GB10 tolerates over-subscription; the 3090 does not.** `config.toml:54` records
  the 3090 *regressing* at 64. Here 128 concurrent requests perform like 48 — you gain
  nothing, but you are not punished either, so the setting is not a cliff on a Spark.
* **Output was byte-stable across settings** (257,489–257,491 chars over 128 pages, 0
  empty), which is the control: the timing differences are concurrency and nothing else.

⚠️ **Do not read a sweep's argmax as its answer.** The script first reported
*"best: parallel=128"* — a true argmax over a plateau whose spread was 1.4 %, which reads
as "still scaling" and is false. It now reports the **knee** (cheapest setting within 3 %
of peak) instead, which is the number an operator actually needs.

### This Town (badly faded scan) — NOT RUN, and the reason is itself a result

Started under the same protocol and **abandoned during the first setting**. It is recorded
here rather than dropped, because the partial observation matters more than the table would
have:

⚠️ **A faded scan is at least 2.3x slower per page than a clean one at identical settings.**
`parallel=8` on 128 *This Town* pages was still running at **22 minutes** when it was
stopped; the same setting on 128 *Mustee* pages finished in **9.7 minutes**. Extrapolated,
the full 8-setting sweep was tracking 2–2.5 h against Mustee's ~45 min.

The cause is the one this document keeps rediscovering: **generative OCR costs output
tokens, not pages.** *This Town* is ~88 % sub-0.60-confidence pages
([comparison.md](comparison.md)), and on a page it cannot read surya2 churns toward
`SURYA_MAX_TOKENS_FULL_PAGE` (6144) instead of terminating early — the same effect that made
one Spark look 4.7x slower than an identical sibling in §11.

Two practical consequences that do not need the table:

* **Budget whole-book OCR by scan quality, not page count.** A 536-page faded book is not
  "1.5x a 359-page clean book"; on this evidence it is several times the work.
* **`surya2_max_tokens` is the lever for a faded book, not `surya2_parallel`.** Concurrency
  cannot help a page that is expensive because it generates 6k tokens of nothing; lowering
  the cap bounds the damage directly. Worth a sweep of its own (bake-off #10).

**Still open:** whether the *knee* moves on faded input. Nothing here answers that — only
that the whole curve shifts down. If it is ever run, trim to `8/32/64/128`; the Mustee
sweep already established the curve's shape, so four points suffice to locate the knee.

### What it means for placement

Combined with §11: a single GB10 saturates at ~**2.3 s/page** and a second unit buys
1.55x, so the realistic floor for surya2 on Sparks is ~**1.5 s/page across two nodes** —
still ~2.5x slower than PaddleOCR-VL alone on the 3090 (0.60 s/page), which is out on
fidelity, not speed. **Adding Sparks does not change that ordering, and neither does
raising `surya2_parallel`** — which is the strongest argument yet against dedicating whole
Sparks to OCR (§9's recommendation stands: co-resident, discovered via `pool = "auto"`).

## 13. Open questions — the bake-off list

1. ~~**surya2 aggregate pages/min: companion vs 3090 vs a GB10 recipe**, sweeping
   parallel — does the GB10 ever cross the 3090?~~ **ANSWERED §11/§12 (2026-08-06):** a
   GB10 saturates at ~2.3 s/page (knee at parallel=32, flat to 128); two nodes reach
   ~1.5 s/page. It does **not** cross the 3090 on speed. Still open: the *companion*
   slot's own curve, which §10's 8 GB/ctx-12288 note predicts is the tightest of the three.
2. **Companion's real concurrency ceiling** — ctx 12288 / KV-slot arithmetic vs
   `parallel=32`: how much of the fan-out actually queues? (The §12 sweep covered a GB10
   only; the 8 GB slot is the case most likely to be over-subscribed.)
3. **vl32 seconds/page on the dedicated 3090** — retire or confirm the 3–4 min/page
   CPU-offload-era number.
4. **Cleanup three-way** on the gold set: verifier hold rate + XHTML validity + tok/s,
   DS4-Flash vs qwen35-122b vs qwen2.5-coder:32b.
5. **Consensus economics**: `agree_floor` sweep — how many low-conf pages get promoted
   without fidelity loss, at what VLM cost per book?
6. **OCR-mode round-trip wall clock** — both 3090 flips, cookbook apply end to end.
7. **Does a sparkrun-compatible Qwen3-VL (ideally MoE) build exist**, and what is the real
   staging time to spark4?
8. **Can LLMConfig round-robin one served name across lanes?** That would unblock 4-Spark
   sharding with zero client changes.
10. **`surya2_max_tokens` on a faded scan** — the §12 abort says a faded book is >=2.3x
    slower per page because unreadable pages churn to the 6144-token cap. Sweep the cap
    (2048/3072/4096/6144) on *This Town* against CER on its gold pages: how much wall clock
    does a lower ceiling buy, and at what page does it start truncating real text?
11. **Does the concurrency knee move on faded input?** The §12 sweep covered a clean scan
    only. Trim to 8/32/64/128 if run.

9. **Chandra-OCR-2 on the gold set** (`epubocr eval`): CER/WER vs surya2 and
   qwen2.5-vl-32b, *especially* on faded pages — does it stay empty rather than fluent on
   unreadable input, and can vLLM logprobs yield a usable confidence proxy? A win here
   upgrades the challenger slot; a decisive win plus a workable conf signal would reopen
   the ground-truth question itself.
