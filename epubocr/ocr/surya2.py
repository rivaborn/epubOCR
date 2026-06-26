"""Surya 2 (surya-ocr >= 0.20) — the optional, served-VLM Surya engine.

Surya 0.20 ("Surya 2") replaced the local detection/recognition/layout models with a
single VLM served through a backend — **vllm** (NVIDIA) or **llama.cpp** (CPU/Apple) —
via ``SuryaInferenceManager``. So it is a *generative* OCR engine: more capable on clean
documents, but it can confabulate (it ships its own blank-region guard) and it needs a
running backend. It is therefore **opt-in** and kept entirely separate from the local
0.17 :class:`~epubocr.ocr.surya_marker.SuryaEngine`, which stays the fidelity-first
default. In the pipeline it behaves like the VLM engine: it reports no per-line
confidence, so the degeneracy guard + consensus do the policing rather than a conf floor.

Install with ``uv sync --extra surya2`` (mutually exclusive with the ``surya`` extra —
both are the same ``surya-ocr`` package at incompatible versions) and provide a backend.
"""
from __future__ import annotations

import importlib.metadata as _md

from pathlib import Path

from .base import OCREngine, OcrResult, OcrWord


def _surya_minor() -> tuple[int, int]:
    parts = _md.version("surya-ocr").split(".")
    return (int(parts[0]), int(parts[1]))


def _poly_to_bbox(polygon) -> tuple[int, int, int, int]:
    if not polygon:
        return (0, 0, 0, 0)
    xs = [float(p[0]) for p in polygon]
    ys = [float(p[1]) for p in polygon]
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


def _block_json(b) -> dict | None:
    """Map a Surya 2 ``BlockOCRResult`` to our ``structure.Block`` JSON (heading/list/table/…)."""
    from ..structure import _DROP, _FIGURE, _TABLE, _block_for, _plain

    label = getattr(b, "label", "Text") or "Text"
    if label in _DROP or getattr(b, "skipped", False):
        return None
    html = getattr(b, "html", "") or ""
    if label in _TABLE and html:                 # keep the model's table HTML verbatim
        return {"type": "table", "text": _plain(html), "html": html}
    if label in _FIGURE:                         # a mislabeled text region -> prose
        label = "Text"
    block = _block_for(label, _plain(html))
    return block.to_json() if block else None


class Surya2Engine(OCREngine):
    name = "surya2"
    wants_preprocess = False  # the VLM reads the raw page

    def __init__(self, backend: str | None = "llamacpp", layout: bool = False):
        self._rec = None
        self._manager = None
        self._backend = backend          # "llamacpp" | "vllm" | None (let Surya autodetect)
        self._layout = layout

    def identity(self) -> str:
        try:
            major, minor = _surya_minor()
            ver = f"{major}.{minor}"
        except Exception:
            ver = "0.20"
        return f"surya2:{ver}:{self._backend or 'auto'}{':layout' if self._layout else ''}"

    def _ensure_loaded(self) -> None:
        if self._rec is not None:
            return
        if _surya_minor() < (0, 20):
            raise RuntimeError(
                f"the 'surya2' engine needs surya-ocr>=0.20 (Surya 2); installed "
                f"{_md.version('surya-ocr')}. Install the 'surya2' extra "
                f"(`uv sync --extra surya2`) — it is mutually exclusive with 'surya'."
            )
        try:
            from surya.inference import SuryaInferenceManager
            from surya.recognition import RecognitionPredictor
            # lazy=True: the backend (and model load) only engages on the first OCR call.
            self._manager = SuryaInferenceManager(method=self._backend)
            self._rec = RecognitionPredictor(self._manager)
        except ImportError as exc:
            raise RuntimeError(
                f"Surya 2 needs a serving backend ('{self._backend}'): {exc}. Install it "
                f"(llama.cpp: `pip install llama-cpp-python`; or run vllm) and make the "
                f"Surya 2 model available. The 3090's vLLM is one option once it's free."
            ) from exc

    def run(self, image_path: Path) -> OcrResult:
        from PIL import Image

        from ..structure import _plain

        self._ensure_loaded()
        img = Image.open(image_path).convert("RGB")
        page = self._rec([img], full_page=True)[0]          # PageOCRResult

        blocks = sorted(getattr(page, "blocks", []), key=lambda b: getattr(b, "reading_order", 0))
        texts: list[str] = []
        words: list[OcrWord] = []
        confs: list[float] = []
        for b in blocks:
            if getattr(b, "skipped", False) or getattr(b, "error", False):
                continue
            txt = _plain(getattr(b, "html", "") or "")
            if not txt:
                continue
            texts.append(txt)
            conf = getattr(b, "confidence", None)            # Surya 2 may not expose one -> None
            words.append(OcrWord(text=txt, bbox=_poly_to_bbox(getattr(b, "polygon", None)), conf=conf))
            if conf is not None:
                confs.append(float(conf))

        meta = {"backend": self._backend, "blocks_n": len(getattr(page, "blocks", []))}
        if self._layout:
            meta["blocks"] = [bj for bj in (_block_json(b) for b in blocks) if bj]

        return OcrResult(
            text="\n".join(texts),
            words=words,
            mean_conf=(sum(confs) / len(confs) if confs else None),
            engine=self.name,
            meta=meta,
        )
