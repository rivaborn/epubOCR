"""Chandra on a DESIGNED sample rather than the whole book.

Three strata, chosen because each answers a question the full book would answer more
slowly and no more clearly:
  * the 4 hand-keyed gold pages          -> absolute CER
  * the 12 pages PaddleOCR degenerated on -> are those pages intrinsically hard, or was
                                             that paddle-specific? (the real question)
  * 14 evenly-spaced pages                -> a fair read on ordinary prose
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(r"C:\Coding\rivaborn\epubOCR")
S = Path(__file__).parent
sys.path.insert(0, str(REPO))

from epubocr.config import Endpoint  # noqa: E402
from epubocr.consensus import agreement  # noqa: E402
from epubocr.eval.metrics import cer, is_degenerate, repetition_ratio  # noqa: E402
from epubocr.llm.client import LLMClient  # noqa: E402
from epubocr.ocr.chandra import ChandraEngine  # noqa: E402
from epubocr.ocr.pool import PoolEngine  # noqa: E402

PAGES = REPO / "book_projects" / "The_Mustee_By_Lance_Horner_-_Unicorn_Books" / "extracted" / "pages"
GOLD = json.loads((S / "mustee.gold.json").read_text(encoding="utf-8"))
PADDLE_BAD = [31, 50, 90, 135, 157, 177, 217, 246, 249, 298, 306, 353]
SPREAD = list(range(10, 359, 25))

_FOLD = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
         "\u2013": "-", "\u2014": "-", "\u2026": "..."}


def fold(s: str) -> str:
    for k, v in _FOLD.items():
        s = s.replace(k, v)
    return " ".join(s.split())


def load(sub: str, idx: int) -> tuple[str, float | None]:
    f = S / sub / f"page_{idx + 1:04d}.raw.json"
    if not f.exists():
        return "", None
    d = json.loads(f.read_text(encoding="utf-8"))
    return d.get("text") or "", d.get("mean_conf")


def main() -> None:
    idxs = sorted({int(k) for k in GOLD} | set(PADDLE_BAD) | set(SPREAD))
    paths = [PAGES / f"page_{i + 1:04d}.jpeg" for i in idxs]
    print(f"chandra sample: {len(idxs)} pages across spark2+spark3\n", flush=True)

    def mk(url: str) -> ChandraEngine:
        return ChandraEngine(LLMClient(Endpoint("c", url, "EMPTY")), "chandra-ocr-2",
                             logprobs=True, parallel=16)

    pool = PoolEngine(members=[("spark2", mk("http://192.168.1.51:8001/v1")),
                               ("spark3", mk("http://192.168.1.52:8001/v1"))], chunk=8)
    t0 = time.perf_counter()
    res = pool.run_batch(paths)
    wall = time.perf_counter() - t0
    print(f"wall {wall:.0f}s for {len(paths)} pages -> {wall/len(paths):.2f}s/page")
    print(f"units: {pool.summary()}\n", flush=True)

    out = {}
    for i, r in zip(idxs, res):
        out[i] = {"text": r.text, "conf": r.mean_conf,
                  "rep": round(repetition_ratio(r.text), 3),
                  "degen": is_degenerate(r.text)}
    (S / "chandra_sample.json").write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                           encoding="utf-8")

    print("=" * 72)
    print("GOLD PAGES")
    print("=" * 72)
    cers = []
    for k, g in GOLD.items():
        i = int(k)
        c = cer(fold(g), fold(out[i]["text"]))
        cers.append(c)
        cf = out[i]["conf"]
        print(f"p{i}: CER={c:.4f} conf={cf if cf is None else round(cf,3)} "
              f"chars={len(out[i]['text'])}")
    print(f"mean CER = {sum(cers)/len(cers):.4f}   "
          f"(surya2 0.0018 | paddleocr 0.0035 | surya0.17 0.2507)")

    print()
    print("=" * 72)
    print("THE 12 PAGES PADDLE DEGENERATED ON")
    print("=" * 72)
    print(f"{'page':>5} {'chandra':>8} {'surya2':>8} {'paddle':>8} {'conf':>6} {'rep':>6} "
          f"{'agree(ch,sy2)':>14}")
    n_over = 0
    for i in PADDLE_BAD:
        ch = out[i]["text"]
        sy2, _ = load("ocr_surya2", i)
        pad, _ = load("ocr_paddleocr", i)
        ag = agreement(ch, sy2)
        over = len(ch) > 2 * len(sy2) if sy2 else False
        n_over += over
        cf = out[i]["conf"]
        print(f"{i:>5} {len(ch):>8} {len(sy2):>8} {len(pad):>8} "
              f"{cf if cf is None else round(cf,3):>6} {out[i]['rep']:>6} {ag:>14.3f}"
              + ("   OVER-PRODUCED" if over else ""))
    print(f"\nchandra over-produced on {n_over}/12 of paddle's failure pages")

    print()
    print("=" * 72)
    print("SPREAD PAGES — agreement with surya2")
    print("=" * 72)
    ags = []
    for i in SPREAD:
        sy2, _ = load("ocr_surya2", i)
        if sy2:
            ags.append(agreement(out[i]["text"], sy2))
    if ags:
        lo = sum(1 for a in ags if a < 0.85)
        print(f"n={len(ags)} mean={sum(ags)/len(ags):.3f} "
              f"min={min(ags):.3f} disagree(<0.85)={lo}")


if __name__ == "__main__":
    main()
