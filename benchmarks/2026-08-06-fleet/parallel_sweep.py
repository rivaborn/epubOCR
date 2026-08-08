"""Does surya2_parallel keep scaling past 32 on a GB10?

Sweeps SURYA_INFERENCE_PARALLEL against spark3 (surya-ocr-2 the ONLY resident model, so
nothing else contends) over a fixed page sample, on two books with opposite character:

  * The Mustee  — clean 1968 hardback; pages generate ~2.2k chars each
  * This Town   — badly faded scan (mean conf 0.49 on surya 0.17); generation length,
                  and therefore the shape of the concurrency curve, may differ entirely

Method notes that matter for believing the numbers:
  * the SAME pages in the SAME order for every setting, images decoded ONCE up front and
    excluded from the timing
  * SURYA_INFERENCE_PARALLEL is read by the backend at generate() time, so one engine
    instance can be re-tuned between runs — no reconnect, no reload, no cold-start bias
  * a warm-up batch runs before the sweep so the first real setting is not charged for it
  * sample >= max parallel, or the setting could not be exercised
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(r"C:\Coding\rivaborn\epubOCR")
S = Path(__file__).parent
sys.path.insert(0, str(REPO))

from epubocr.ocr.surya2 import Surya2Engine  # noqa: E402

SPARK3 = "http://192.168.1.52:8000/v1"
PARALLELS = [8, 16, 24, 32, 48, 64, 96, 128]
SAMPLE = 128

BOOKS = {
    "The_Mustee": REPO / "book_projects" / "The_Mustee_By_Lance_Horner_-_Unicorn_Books",
    "This_Town": REPO / "book_projects" / "This_Town",
}


def sample_images(project: Path, n: int):
    from PIL import Image

    manifest = json.loads((project / "extracted" / "manifest.json").read_text(encoding="utf-8"))
    pages = [p for p in manifest["pages"] if p["type"] in ("image", "cover")]
    step = max(1, len(pages) // n)
    picked = pages[::step][:n]
    imgs, names = [], []
    for p in picked:
        f = (p.get("extracted_images") or [None])[0]
        if not f:
            continue
        path = project / "extracted" / "pages" / f
        if not path.exists():
            continue
        imgs.append(Image.open(path).convert("RGB"))
        names.append(f)
    return imgs, names


def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    results: dict = {}
    out_path = S / "parallel_sweep.json"
    if out_path.exists():
        results = json.loads(out_path.read_text(encoding="utf-8"))

    eng = Surya2Engine(backend="vllm", inference_url=SPARK3, model="surya-ocr-2", parallel=8)
    eng._ensure_loaded()
    from surya.settings import settings as sset

    for book, project in BOOKS.items():
        if only and book != only:
            continue
        imgs, names = sample_images(project, SAMPLE)
        print(f"\n{'='*76}\n{book}: {len(imgs)} pages, spark3 (surya-ocr-2 only)\n{'='*76}",
              flush=True)

        print("warm-up (8 pages)...", flush=True)
        sset.SURYA_INFERENCE_PARALLEL = 8
        eng._rec(imgs[:8], full_page=True)

        rows = results.setdefault(book, {})
        print(f"{'parallel':>8} {'wall_s':>8} {'s/page':>8} {'pages/s':>8} {'vs p8':>7} "
              f"{'out_chars':>10} {'empty':>6}", flush=True)
        base = None
        for par in PARALLELS:
            if par > len(imgs):
                print(f"{par:>8}   SKIPPED (sample {len(imgs)} < parallel)", flush=True)
                continue
            sset.SURYA_INFERENCE_PARALLEL = par
            t0 = time.perf_counter()
            pages = eng._rec(imgs, full_page=True)
            wall = time.perf_counter() - t0
            res = [eng._page_to_result(p) for p in pages]
            chars = sum(len(r.text) for r in res)
            empty = sum(1 for r in res if not r.text.strip())
            base = base or wall
            rows[str(par)] = {"wall": round(wall, 1), "s_per_page": round(wall / len(imgs), 3),
                              "pages_per_s": round(len(imgs) / wall, 3), "chars": chars,
                              "empty": empty, "n": len(imgs)}
            print(f"{par:>8} {wall:>8.1f} {wall/len(imgs):>8.3f} {len(imgs)/wall:>8.2f} "
                  f"{base/wall:>6.2f}x {chars:>10} {empty:>6}", flush=True)
            out_path.write_text(json.dumps(results, indent=1), encoding="utf-8")

        # Report the KNEE, not the argmax. The argmax over a plateau is whichever run
        # happened to catch a good second — on The Mustee it crowned parallel=128 over a
        # 48-128 band whose spread was 1.4%, which reads as "keeps scaling" and is false.
        # The useful number is the cheapest setting within noise of the best.
        best = max(rows.items(), key=lambda kv: kv[1]["pages_per_s"])
        best_rate = best[1]["pages_per_s"]
        knee = min((int(k) for k, v in rows.items() if v["pages_per_s"] >= 0.97 * best_rate),
                   default=int(best[0]))
        spread = ((best_rate - min(v["pages_per_s"] for v in rows.values())) / best_rate) * 100
        print(f"\npeak {best_rate:.3f} pages/s at parallel={best[0]}; "
              f"KNEE (within 3% of peak) = parallel={knee}", flush=True)
        print(f"range across all settings: {spread:.0f}% slower at the worst", flush=True)
        for im in imgs:
            im.close()


if __name__ == "__main__":
    main()
