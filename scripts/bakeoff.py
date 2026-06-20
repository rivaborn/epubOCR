"""Preprocessing / engine bake-off on a sample of pages.

Surya (raw / +contrast / +binarize-CLAHE) over a spread of image pages, plus a VLM
(qwen2.5-vl-32b on vLLM) over a smaller subset (it's slow: 32B + CPU-offload). Scores
per-engine **confidence** and cross-engine **consensus** (agreement with the VLM).
Self-contained: writes only to qa/bakeoff*, never the main ocr/.

Run:  .venv/Scripts/python.exe scripts/bakeoff.py <book-slug> [n_surya] [n_vlm]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

from epubocr.config import load_config
from epubocr.consensus import agreement
from epubocr.eval import metrics
from epubocr.llm.client import LLMClient
from epubocr.ocr.surya_marker import SuryaEngine
from epubocr.ocr.vlm_openai import VlmOcrEngine
from epubocr.preprocess import preprocess_image
from epubocr.storage import BookProject

CONTRAST_CFG = {"autocontrast_cutoff": 0, "contrast": 1.3}  # gentle: lift without blowing out faint text
BINARIZE_CFG = {"clahe": True, "clahe_clip": 2.0, "binarize": "adaptive",
                "adaptive_block": 31, "adaptive_c": 15, "deskew": True}
VLM_MODEL = "qwen2.5-vl-32b"
VLM_MAX_SIDE = 1400  # downscale for the VLM to keep 32B latency sane


def sample_pages(manifest: dict, n: int) -> list[dict]:
    imgs = [p for p in manifest["pages"] if p["type"] == "image" and p.get("extracted_images")]
    if len(imgs) <= n:
        return imgs
    step = len(imgs) / n
    return [imgs[int(i * step)] for i in range(n)]


def _downscale(src: Path, dst: Path, max_side: int) -> Path:
    im = Image.open(src)
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side))
    im.save(dst)
    return dst


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main():
    slug = sys.argv[1] if len(sys.argv) > 1 else "This_Town"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    n_vlm = int(sys.argv[3]) if len(sys.argv) > 3 else n
    vlm_endpoint = sys.argv[4] if len(sys.argv) > 4 else "ollama"       # fast 7B by default
    vlm_model = sys.argv[5] if len(sys.argv) > 5 else "qwen2.5vl:7b"

    cfg = load_config()
    project = BookProject(root=cfg.projects_root / slug)
    manifest = project.read_json(project.manifest_path)
    pages = sample_pages(manifest, n)
    if n_vlm > 0:
        step = max(1, len(pages) // n_vlm)
        vlm_idx = set(range(0, len(pages), step))     # spread the VLM subset across the book
    else:
        vlm_idx = set()
    tmp = project.qa / "bakeoff_tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    surya = SuryaEngine()
    vlm = VlmOcrEngine(LLMClient(cfg.endpoint(vlm_endpoint)), vlm_model)

    rows = []
    print(f"bake-off: {len(pages)} pages (Surya x3), VLM on {len(vlm_idx)} of them", flush=True)
    for k, page in enumerate(pages):
        idx = page["index"]
        src = project.pages / page["extracted_images"][0]
        s_raw = surya.run(src)
        s_con = surya.run(preprocess_image(src, tmp / f"{idx}_contrast.png", CONTRAST_CFG))
        s_bin = surya.run(preprocess_image(src, tmp / f"{idx}_binarize.png", BINARIZE_CFG))

        v_text = None
        if k in vlm_idx:
            v = vlm.run(_downscale(src, tmp / f"{idx}_vlm.png", VLM_MAX_SIDE))
            v_text = "" if metrics.is_degenerate(v.text) else v.text

        row = {"index": idx,
               "raw": {"conf": s_raw.mean_conf}, "contrast": {"conf": s_con.mean_conf},
               "binarize": {"conf": s_bin.mean_conf}, "has_vlm": v_text is not None,
               "vlm_degenerate": v_text == ""}
        if v_text is not None:
            row["raw"]["agree"] = agreement(s_raw.text, v_text)
            row["contrast"]["agree"] = agreement(s_con.text, v_text)
            row["binarize"]["agree"] = agreement(s_bin.text, v_text)
        rows.append(row)
        msg = (f"  [{k+1}/{len(pages)}] p{idx}: raw={_f(s_raw.mean_conf)} "
               f"con={_f(s_con.mean_conf)} bin={_f(s_bin.mean_conf)}")
        if v_text is not None:
            msg += f"  agree(bin,vlm)={row['binarize']['agree']:.2f}"
        print(msg, flush=True)

    project.write_json(project.qa / "bakeoff.json",
                       {"model_vlm": vlm_model, "n": len(pages), "n_vlm": len(vlm_idx), "rows": rows})
    _report(rows, vlm_model)


def _f(x):
    return f"{x:.2f}" if x is not None else "n/a"


def _report(rows, vlm_name):
    n = len(rows)
    vlm_rows = [r for r in rows if r["has_vlm"]]
    print(f"\n=== BAKE-OFF SUMMARY === Surya x{n} pages, VLM x{len(vlm_rows)} ({vlm_name})")
    print("variant   | mean_conf | >=0.80 | mean_agree(vlm) | trusted")
    print("----------+-----------+--------+-----------------+--------")
    for key in ("raw", "contrast", "binarize"):
        confs = [r[key]["conf"] for r in rows]
        hi = sum(1 for c in confs if c is not None and c >= 0.80)
        agrs = [r[key].get("agree") for r in vlm_rows]
        # consensus trust over the FULL sample: confident OR (where measured) agrees with VLM
        trusted = sum(1 for r in rows
                      if (r[key]["conf"] is not None and r[key]["conf"] >= 0.80)
                      or (r.get("has_vlm") and r[key].get("agree", 0) >= 0.85))
        print(f"surya-{key:4s}| {_f(mean(confs)):^9} | {hi:>4}   | {_f(mean(agrs)):^15} | {trusted:>3}/{n}")
    degen = sum(1 for r in vlm_rows if r["vlm_degenerate"])
    print(f"\nvlm32 degenerate: {degen}/{len(vlm_rows)}")
    print("trusted = pages renderable as reflowable (Surya conf>=0.80 OR agrees with VLM>=0.85)")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
