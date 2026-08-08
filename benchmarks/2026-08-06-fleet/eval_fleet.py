"""Fleet OCR accuracy bake-off: surya2 + chandra (spark3) vs paddleocr-vl (3070 Ti).

Scores against a hand-transcribed gold set from The_Mustee (real 1965 scan).
Reports raw CER/WER/insertion plus a punctuation-folded CER (curly quotes and
dashes normalized) so typographic-convention noise doesn't mask real differences.
Chandra additionally gets a markdown-stripped score (it may emit md formatting).
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

REPO = Path(r"C:\Coding\rivaborn\epubOCR")
SCRATCH = Path(__file__).parent
sys.path.insert(0, str(REPO))

from epubocr.config import Endpoint  # noqa: E402
from epubocr.eval.metrics import cer, wer, insertion_rate  # noqa: E402
from epubocr.llm.client import LLMClient  # noqa: E402
from epubocr.ocr.surya2 import Surya2Engine  # noqa: E402
from epubocr.ocr.vlm_openai import VlmOcrEngine  # noqa: E402

PROJECT = REPO / "book_projects" / "The_Mustee_By_Lance_Horner_-_Unicorn_Books"
PAGES = PROJECT / "extracted" / "pages"
GOLD = json.loads((SCRATCH / "mustee.gold.json").read_text(encoding="utf-8"))
# manifest index -> page image / cached surya-0.17 raw.json stem
IMG = {4: "page_0005", 5: "page_0006", 6: "page_0007", 7: "page_0008"}

_FOLD = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2026": "...",
}


def fold(s: str) -> str:
    for k, v in _FOLD.items():
        s = s.replace(k, v)
    return s


def strip_md(s: str) -> str:
    s = re.sub(r"```.*?```", "", s, flags=re.S)          # code fences
    s = re.sub(r"</?[a-zA-Z][^>]*>", "", s)              # html tags
    s = re.sub(r"^#{1,6}\s*", "", s, flags=re.M)         # headings
    s = re.sub(r"[*_]{1,3}(\S(?:.*?\S)?)[*_]{1,3}", r"\1", s)  # bold/italic
    s = re.sub(r"^\s*[-|:]+\s*$", "", s, flags=re.M)     # table rules
    s = s.replace("|", " ")
    return s


def score(gold: str, text: str) -> dict:
    return {
        "cer": round(cer(gold, text), 4),
        "wer": round(wer(gold, text), 4),
        "insertion": round(insertion_rate(gold, text), 4),
        "cer_folded": round(cer(fold(gold), fold(text)), 4),
    }


def main() -> None:
    engines = {
        "surya2@spark3": Surya2Engine(
            backend="vllm", inference_url="http://192.168.1.52:8000/v1",
            model="surya-ocr-2", parallel=8),
        "chandra@spark3": VlmOcrEngine(
            LLMClient(Endpoint("chandra", "http://192.168.1.52:8001/v1", "EMPTY")),
            "chandra-ocr-2"),
        "paddle@3070Ti": VlmOcrEngine(
            LLMClient(Endpoint("paddle",
                               "http://192.168.1.40:11430/lane/companion/v1", "EMPTY")),
            "paddleocr-vl-1.6"),
    }

    results: dict = {}
    for name, eng in engines.items():
        results[name] = {}
        for idx, stem in IMG.items():
            img = PAGES / f"{stem}.jpeg"
            t0 = time.perf_counter()
            try:
                r = eng.run(img)
                dt = time.perf_counter() - t0
                entry = {
                    "text": r.text, "mean_conf": r.mean_conf,
                    "secs": round(dt, 1), **score(GOLD[str(idx)], r.text),
                }
                if name.startswith("chandra"):
                    entry["cer_md_stripped"] = round(
                        cer(fold(GOLD[str(idx)]), fold(strip_md(r.text))), 4)
            except Exception as exc:  # noqa: BLE001
                entry = {"error": f"{type(exc).__name__}: {exc}"}
            results[name][idx] = entry
            e = results[name][idx]
            print(f"[{name}] p{idx}: "
                  + (f"CER={e['cer']} folded={e['cer_folded']} conf={e.get('mean_conf')} "
                     f"{e['secs']}s ({len(e['text'])} ch)"
                     if "error" not in e else e["error"]), flush=True)

    # bonus row: cached surya 0.17 output (engine 'surya', from ocr/*.raw.json)
    results["surya0.17@cached"] = {}
    for idx, stem in IMG.items():
        raw = json.loads((PROJECT / "ocr" / f"{stem}.raw.json").read_text(encoding="utf-8"))
        text = raw.get("text") or ""
        results["surya0.17@cached"][idx] = {
            "text": text, "mean_conf": raw.get("mean_conf"), "secs": None,
            **score(GOLD[str(idx)], text),
        }

    (SCRATCH / "fleet_eval_results.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")

    print("\nengine           | page | CER    | CERfold | WER    | insert | conf  | secs")
    print("-" * 78)
    for name, pages in results.items():
        for idx, e in pages.items():
            if "error" in e:
                print(f"{name:16s} | {idx:>4} | ERROR: {e['error'][:50]}")
                continue
            conf = f"{e['mean_conf']:.2f}" if e.get("mean_conf") is not None else " n/a"
            secs = f"{e['secs']}" if e.get("secs") is not None else "cache"
            print(f"{name:16s} | {idx:>4} | {e['cer']:.4f} | {e['cer_folded']:.4f}  "
                  f"| {e['wer']:.4f} | {e['insertion']:.4f} | {conf} | {secs}")


if __name__ == "__main__":
    main()
