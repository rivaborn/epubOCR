"""Score the per-engine whole-book OCR runs.

With only 4 hand-keyed gold pages, book-scale accuracy is measured three ways:
  1. CER/WER on the gold pages (absolute, small n)
  2. pairwise cross-engine agreement over EVERY page (model-agnostic; the signal
     comparison.md already validated) — disagreement localizes the hard pages
  3. distributional tells: confidence, empty pages, degenerate pages, and
     over-production (a page where one engine emits far more text than the others is
     the hallucination signature this pipeline exists to catch)
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO = Path(r"C:\Coding\rivaborn\epubOCR")
SCRATCH = Path(__file__).parent
sys.path.insert(0, str(REPO))

from epubocr.consensus import agreement  # noqa: E402
from epubocr.eval.metrics import cer, wer  # noqa: E402

GOLD = json.loads((SCRATCH / "mustee.gold.json").read_text(encoding="utf-8"))
_FOLD = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
         "\u2013": "-", "\u2014": "-", "\u2026": "..."}


def fold(s: str) -> str:
    for k, v in _FOLD.items():
        s = s.replace(k, v)
    return " ".join(s.split())


def load_run(d: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for f in sorted(d.glob("page_*.raw.json")):
        idx = int(f.stem.split("_")[1].split(".")[0]) - 1     # page_0001 -> index 0
        j = json.loads(f.read_text(encoding="utf-8"))
        out[idx] = {"text": j.get("text") or "", "conf": j.get("mean_conf"),
                    "engine": j.get("engine"), "degenerate": j.get("degenerate"),
                    "unit": (j.get("meta") or {}).get("unit")}
    return out


def main() -> None:
    runs = {}
    for name, sub in [("surya2", "ocr_surya2"), ("chandra", "ocr_chandra"),
                      ("paddleocr", "ocr_paddleocr"), ("surya0.17", "ocr_baseline_surya017")]:
        d = SCRATCH / sub
        if d.exists():
            runs[name] = load_run(d)
    if not runs:
        print("no runs found"); return

    print("=" * 74)
    print("1. GOLD-PAGE ACCURACY (hand-transcribed, n=4: 3 prose + 1 blank)")
    print("=" * 74)
    print(f"{'engine':12s} | {'page':>4} | {'CER':>7} | {'WER':>7} | {'conf':>5} | chars")
    print("-" * 74)
    for name, pages in runs.items():
        cers = []
        for k, gold in GOLD.items():
            idx = int(k)
            p = pages.get(idx)
            if p is None:
                continue
            c = cer(fold(gold), fold(p["text"]))
            cers.append(c)
            cf = f"{p['conf']:.2f}" if p["conf"] is not None else "  n/a"
            print(f"{name:12s} | {idx:>4} | {c:>7.4f} | "
                  f"{wer(fold(gold), fold(p['text'])):>7.4f} | {cf:>5} | {len(p['text'])}")
        if cers:
            print(f"{name:12s} |  ALL | {statistics.mean(cers):>7.4f} | "
                  f"{'':>7} | {'':>5} | mean CER")
        print("-" * 74)

    print()
    print("=" * 74)
    print("2. WHOLE-BOOK DISTRIBUTION")
    print("=" * 74)
    print(f"{'engine':12s} | pages | empty | degen | mean conf | conf<0.80 | mean chars")
    print("-" * 74)
    for name, pages in runs.items():
        n = len(pages)
        empty = sum(1 for p in pages.values() if not p["text"].strip())
        degen = sum(1 for p in pages.values() if p["degenerate"])
        confs = [p["conf"] for p in pages.values() if p["conf"] is not None]
        low = sum(1 for c in confs if c < 0.80)
        chars = [len(p["text"]) for p in pages.values()]
        mc = f"{statistics.mean(confs):.3f}" if confs else "  n/a"
        print(f"{name:12s} | {n:>5} | {empty:>5} | {degen:>5} | {mc:>9} | "
              f"{low:>9} | {statistics.mean(chars):>10.0f}")

    print()
    print("=" * 74)
    print("3. CROSS-ENGINE AGREEMENT (every shared page)")
    print("=" * 74)
    names = [n for n in ("surya2", "chandra", "paddleocr", "surya0.17") if n in runs]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = sorted(set(runs[a]) & set(runs[b]))
            ags = [agreement(runs[a][k]["text"], runs[b][k]["text"]) for k in shared]
            if not ags:
                continue
            lo = sum(1 for x in ags if x < 0.85)
            print(f"{a:10s} vs {b:10s} n={len(ags):>4}  mean={statistics.mean(ags):.3f}  "
                  f"median={statistics.median(ags):.3f}  disagree(<0.85)={lo} "
                  f"({100*lo/len(ags):.0f}%)")

    print()
    print("=" * 74)
    print("4. OVER-PRODUCTION — the hallucination signature")
    print("=" * 74)
    print("pages where one engine emits >2x the median length of the others")
    if len(names) >= 2:
        allpages = sorted(set.intersection(*[set(runs[n]) for n in names]))
        flags: dict[str, list[int]] = {n: [] for n in names}
        for k in allpages:
            lens = {n: len(runs[n][k]["text"]) for n in names}
            med = statistics.median(lens.values())
            if med < 50:                     # near-blank page: absolute test instead
                for n, L in lens.items():
                    if L > 200:
                        flags[n].append(k)
                continue
            for n, L in lens.items():
                if L > 2 * med:
                    flags[n].append(k)
        for n in names:
            ex = ", ".join(str(x) for x in flags[n][:12])
            print(f"{n:12s} {len(flags[n]):>4} page(s){'  e.g. ' + ex if ex else ''}")

    (SCRATCH / "score_summary.json").write_text(json.dumps(
        {n: {"pages": len(p),
             "empty": sum(1 for x in p.values() if not x["text"].strip()),
             "mean_conf": (statistics.mean([x["conf"] for x in p.values()
                                            if x["conf"] is not None])
                           if any(x["conf"] is not None for x in p.values()) else None)}
         for n, p in runs.items()}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
