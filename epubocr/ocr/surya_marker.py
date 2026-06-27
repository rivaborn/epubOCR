"""Surya OCR engine — fidelity-grade detection + recognition + reading order.

The traditional ground-truth engine in the fidelity-first policy (epubOCR.md §4):
deterministic transcription with **per-line confidence** (so the builder can route
low-confidence pages to facsimile) and built-in repeated-text dropping (a second
guard against the degenerate loops VLMs fall into).

Predictors are loaded once and reused across pages. First use downloads the Surya
models (~1-2 GB). Install with ``uv sync --extra surya``.
"""
from __future__ import annotations

from pathlib import Path

from .base import OCREngine, OcrResult, OcrWord


class SuryaEngine(OCREngine):
    name = "surya"
    wants_preprocess = False  # Surya does its own detection/normalization

    def __init__(self, device: str | None = None, dtype=None, layout: bool = False):
        self._rec = None
        self._det = None
        self._layout_pred = None
        self._device = device
        self._dtype = dtype
        self._layout = layout            # opt-in: also produce structured blocks (headings/tables/order)

    def identity(self) -> str:
        return f"surya:0.17:{self._device or 'auto'}{':layout' if self._layout else ''}"

    def _ensure_loaded(self) -> None:
        if self._rec is not None:
            return
        import importlib.metadata as _md
        _ver = _md.version("surya-ocr")
        if tuple(int(x) for x in _ver.split(".")[:2]) >= (0, 18):
            raise RuntimeError(
                f"the 'surya' engine needs surya-ocr 0.17.x (local models); installed {_ver}. "
                f"Install the 'surya' extra (`uv sync --extra surya`), or use engine 'surya2' "
                f"for Surya 2 (the served-VLM architecture, >=0.20)."
            )
        import torch
        from surya.detection import DetectionPredictor
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = self._dtype or (torch.float16 if device == "cuda" else torch.float32)
        self._device = device
        foundation = FoundationPredictor(device=device, dtype=dtype)   # shared across predictors
        self._rec = RecognitionPredictor(foundation)
        self._det = DetectionPredictor(device=device, dtype=dtype)
        if self._layout:
            from surya.layout import LayoutPredictor
            self._layout_pred = LayoutPredictor(foundation)

    def run(self, image_path: Path) -> OcrResult:
        return self.run_batch([image_path])[0]

    def run_batch(self, image_paths: list[Path]) -> list[OcrResult]:
        """OCR a chunk of pages in one on-GPU batch (Surya batches recognition over
        the whole list), then layout once over the same list when enabled."""
        from PIL import Image

        self._ensure_loaded()
        images = [Image.open(p).convert("RGB") for p in image_paths]
        # sort_lines -> reading order; drop_repeated_text -> kill degenerate loops;
        # math_mode off -> plain prose, no spurious LaTeX wrapping.
        results = self._rec(images, det_predictor=self._det, sort_lines=True,
                            drop_repeated_text=True, math_mode=False)
        layouts = self._layout_pred(images) if self._layout_pred is not None else [None] * len(images)
        return [self._to_ocr(res, lay) for res, lay in zip(results, layouts)]

    def _to_ocr(self, result, layout_res) -> OcrResult:
        from ..structure import build_blocks, poly_bbox

        texts: list[str] = []
        words: list[OcrWord] = []
        confs: list[float] = []
        for line in result.text_lines:
            txt = (line.text or "").strip()
            texts.append(txt)
            conf = getattr(line, "confidence", None)
            words.append(OcrWord(text=txt, bbox=_poly_to_bbox(getattr(line, "polygon", None)),
                                 conf=conf))
            if conf is not None:
                confs.append(float(conf))

        meta = {"device": self._device, "lines": len(result.text_lines)}
        if layout_res is not None:
            line_items = [((ln.text or ""), poly_bbox(ln.polygon)) for ln in result.text_lines]
            blocks = build_blocks(line_items, layout_res.bboxes)
            meta["blocks"] = [b.to_json() for b in blocks]

        return OcrResult(
            text="\n".join(texts),
            words=words,
            mean_conf=(sum(confs) / len(confs) if confs else None),
            engine=self.name,
            meta=meta,
        )


def _poly_to_bbox(polygon) -> tuple[int, int, int, int]:
    if not polygon:
        return (0, 0, 0, 0)
    xs = [float(p[0]) for p in polygon]
    ys = [float(p[1]) for p in polygon]
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
